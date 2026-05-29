import pickle
import torch

from utils import set_random_state
from models.tmg_gan import TMGGAN
from classifier import Classifier
from datasets.tr_dataset import TrDataset
from datasets.te_dataset import TeDataset
from datasets.data import tr_samples, tr_labels, te_labels, te_samples, feature_num, label_num


if __name__ == '__main__':

    seed = 888
    set_random_state(seed)
    tmg_gan = TMGGAN(feature_num, label_num)
    tmg_gan.fit(TrDataset())
    # count the max number of samples
    max_cnt = max([len(tmg_gan.samples[i]) for i in tmg_gan.samples.keys()])
    # generate samples
    for i in tmg_gan.samples.keys():
        cnt_generated = max_cnt - len(tmg_gan.samples[i])
        if cnt_generated > 128:
            print(f"cnt_generated is {cnt_generated}")
            generated_samples = tmg_gan.generate_qualified_samples(i, cnt_generated)
            generated_labels = torch.full([cnt_generated], i)
            tr_samples = torch.cat([tr_samples, generated_samples])
            tr_labels = torch.cat([tr_labels, generated_labels])

    with open('data.pkl', 'wb') as f:
        pickle.dump(
            (
                tr_samples.numpy(),
                tr_labels.numpy(),
                te_samples.numpy(),
                te_labels.numpy(),
            ),
            f,
        )

    clf = Classifier('TMG_GAN', feature_num, label_num)
    clf.model = tmg_gan.cd
    clf.fit(TrDataset())
    torch.cuda.empty_cache()
    clf.test(TeDataset())
    print(clf.confusion_matrix)
    print(f'Accuracy: {clf.metrics["Accuracy"]:.4f}')
    print(f'Precision: {clf.metrics["Precision"]:.4f}')
    print(f'Recall: {clf.metrics["Recall"]:.4f}')
    print(f'F1: {clf.metrics["F1"]:.4f}')

import datetime
import sys
import os
import time
import torch
import torch.utils.data as Data
from tqdm import tqdm
from utils.dataloader_pretrain import ViTDataSet
from torchvision import transforms
from models.model_MTMAE import encoder, decoder
import math

batch_size = 512
lr = 0.0001
lr_decay = 0.05
Epochs = 300
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mask_ratio = 0.6
embed_dim = 128
encoder_depth = 8
decoder_depth = 4


save_model = "./weights"
save_dir_run = "./runs"
if not os.path.exists(save_model):
    os.makedirs(save_model)

if not os.path.exists(save_dir_run):
    os.makedirs(save_dir_run)


def train(train_path):
    mean = [0.5]
    std = [0.5]

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = ViTDataSet(train_path, transform)
    data_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    Encoder = encoder(img_size=28, patch_size=4, in_chans=1, embed_dim=embed_dim,
                      encoder_depth=8, num_heads=8, mlp_ratio=3.).to(device)

    Decoder = decoder(img_size=28, patch_size=4, in_chans=1, embed_dim=embed_dim,
                      decoder_depth=4, num_heads=8, mlp_ratio=3.).to(device)

    # encode/decode optimizers
    optim_e = torch.optim.AdamW(Encoder.parameters(), lr=lr, weight_decay=0.05)
    optim_d = torch.optim.AdamW(Decoder.parameters(), lr=lr, weight_decay=0.05)

    def cosine_decay_scheduler(optimizer, step, total_steps, initial_lr):
        lr = 0.5 * initial_lr * (1 + math.cos(math.pi * step / total_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    #  Loss Func
    criterion = torch.nn.MSELoss().to(device)

    print(f"Start training for {Epochs} epochs")
    start_time = time.time()

    for epoch in range(Epochs):

        Encoder.train()
        Decoder.train()
        # Compute and update learning rate using cosine_decay_scheduler
        cosine_decay_scheduler(optim_e, epoch, Epochs, lr)
        cosine_decay_scheduler(optim_d, epoch, Epochs, lr)
        data_loader = tqdm(data_loader, desc=f'Processing epoch {epoch:02d}')
        total_loss = 0
        for batch, data in enumerate(data_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            latent, ids_restore = Encoder(images, mask_ratio=mask_ratio)
            predict = Decoder(latent, ids_restore)

            loss = criterion(predict, images)
            data_loader.set_postfix({f'loss': f'{loss.item(): 6.3f}'})
            loss.backward()

            optim_e.step()
            optim_d.step()
            Decoder.zero_grad()
            Encoder.zero_grad()

            # print("epoch", epoch, "batch", batch, "loss:", loss.detach().item())
            total_loss += loss.detach().item()
        print("\nepoch loss:", total_loss / len(train_dataset) * batch_size)
        with open(os.path.join(save_dir_run, "pretrain_loss.txt"), "a") as f:
            f.write(str(total_loss / len(train_dataset) * batch_size) + "\n")

        if (epoch+1) % 100 == 0:
            torch.save(Encoder.state_dict(), os.path.join(save_model, f"encoder_{epoch}.pth"))
            torch.save(Decoder.state_dict(), os.path.join(save_model, f"decoder_{epoch}.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    train_path = r'./datasets/pretrain_dataset.tsv'
    train(train_path)

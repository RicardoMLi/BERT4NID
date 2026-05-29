import click

from ml.utils import train_traffic_classification_cnn_model

# python train_cnn.py -d train_test_data/med/train.parquet -m checkpoints/model.pt -n 4
@click.command()
@click.option(
    "-d",
    "--data_path",
    help="training data dir path containing parquet files",
    required=True,
    default=r'target/traffic_classification/train.parquet'
)
@click.option("-m", "--model_path", help="output model path", required=True, default='./checkpoints/model')
@click.option("-n", "--num_classes", help="number of classes", required=True)
def main(data_path, model_path, num_classes):
    train_traffic_classification_cnn_model(data_path, model_path, int(num_classes))


if __name__ == "__main__":
    main()

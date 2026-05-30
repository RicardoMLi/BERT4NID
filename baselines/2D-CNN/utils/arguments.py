from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict


@dataclass
class StageArguments:
    name: str = field(
        default=None,
        metadata={"help": "stage name"}
    )
    category: str = field(
        default=None,
        metadata={"help": "stage class"}
    )
    src_folder: Path = field(
        default=None,
        metadata={"help": "source folder of current stage"}
    )
    dst_folder: Path = field(
        default=None,
        metadata={"help": "dst folder of current stage"}
    )
    file2folder: bool = field(
        default=False,
        metadata={"help": "make new folder for file while traverse the folder"}
    )
    output_dir: Path = field(
        default=None,
        metadata={"help": "root directory for output"}
    )
    src_file: Path = field(
        default=None
    )
    dst_file: Path = field(
        default=None
    )
    cmd: str = field(
        default=None
    )
    num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    kwargs: Dict = field(
        default=None,
        metadata={"help": "other kw args"}
    )


@dataclass
class PreprocessArguments:
    """
    Arguments Preprocess to pcap files to generate dataset_dict.
    """
    name: str = field(
        default="UTSC-TK2016-Pretrain",
        metadata={"help": "Preprocess dataset_dict name"}
    )
    output_dir: str = field(
        default="./tmp",
        metadata={"help": "root directory for output"}
    )
    dataset_src_root_dir: str = field(
        default="/home/USTC-TK2016",
        metadata={"help": "Source dataset_dict (.pcap) folder directory path"}
    )
    dataset_dst_root_dir: str = field(
        default="./tmp",
        metadata={"help": "Generated dataset_dict folder directory path"}
    )
    split_session_folder: str = field(
        default="split_sessions",
        metadata={"help": "Folder to store the split pcap session"}
    )
    splitcap_path: str = field(
        default="./tools/SplitCap.exe",
        metadata={"help": "Path to splitcap.exe (https://www.netresec.com/?page=SplitCap), "
                          "which split pcap to sessions"}
    )
    trim_time_folder: str = field(
        default="trim_time",
        metadata={"help": "Folder to store the trimmed sessions"}
    )
    time_window: int = field(
        default=3600,
        metadata={"help": "Time length (seconds) of the trim slice"}
    )
    min_packet_num: int = field(
        default=6,
        metadata={"help": "Min packet num (quantity) of the trim slice"}
    )
    min_file_size: int = field(
        default=200,
        metadata={"help": "Min file size"}
    )
    max_file_size: int = field(
        default=784,
        metadata={"help": "Max file size for each burst"}
    )
    njobs: Optional[int] = field(
        default=8,
        metadata={"help": "The number of executors."},
    )
    num_samples_per_class: int = field(
        default=6000,
        metadata={"help": "The number of samples per class to save training time."},
    )
    seed: int = field(
        default=3407,
        metadata={"help": "The random seed number for preprocessing dataset."},
    )
    num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )


@dataclass
class ModelArguments:
    """
    Arguments for model we are going to pretrain.
    """

    image_size: int = field(
        default=28,
        metadata={"help": "Input image size"}
    )
    patch_size: int = field(
        default=2,
        metadata={"help": "The size of patches"}
    )
    hidden_size: Optional[int] = field(
        default=384,
        metadata={"help": "The hidden size for projection layer in transformer encoder."},
    )
    num_layers: Optional[int] = field(
        default=6,
        metadata={"help": "The number of encoder layers in transformer encoder."},
    )
    heads: Optional[int] = field(
        default=6,
        metadata={"help": "The number of heads in multi self attention layer."},
    )
    projector_input_dim: Optional[int] = field(
        default=128,
        metadata={"help": "The dimension of input for projector."},
    )
    projector_dim: Optional[int] = field(
        default=256,
        metadata={"help": "The hidden size for projector."},
    )
    out_ndim: Optional[int] = field(
        default=128,
        metadata={"help": "The dimension of output for projector."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_dir: str = field(
        metadata={"help": "The path for storing dataset."}
    )
    device: str = field(
        default="cuda",
        metadata={"help": "The device for training."},
    )
    test_size: float = field(
        default=0.15,
        metadata={"help": "The test ratio for train test split."},
    )
    seed: int = field(
        default=3407,
        metadata={"help": "The random seed number for training."},
    )
    num_workers: Optional[int] = field(
        default=0,
        metadata={"help": "The number of workers."},
    )
    patience: int = field(
        default=15,
        metadata={"help": "The patience for early stopping."},
    )


@dataclass
class DataPretrainingArguments(DataTrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for pre-training.
    """
    epochs: int = field(
        default=500,
        metadata={"help": "The number of epoch."},
    )
    batch_size: int = field(
        default=512,
        metadata={"help": "The batch size."},
    )
    learning_rate: float = field(
        default=5e-4,
        metadata={"help": "The learning rate."},
    )
    weight_decay: float = field(
        default=0.05,
        metadata={"help": "The weight_decay."},
    )
    nsampling: int = field(
        default=4,
        metadata={"help": "The sampling view number in APS paper."},
    )
    sampling_ratio: float = field(
        default=0.25,
        metadata={"help": "The sampling ratio in APS paper."},
    )
    power: float = field(
        default=3.0,
        metadata={"help": "The sampling power in APS paper."},
    )
    temperature: float = field(
        default=0.1,
        metadata={"help": "The temperature to soften."},
    )
    T_max: int = field(
        default=10,
        metadata={"help": "The t-max value for CosineAnnealingLR."},
    )
    pretrain: bool = field(
        default=True,
        metadata={"help": "Whether to pretrain or fintune."}
    )
    checkpoint_path: str = field(
        default='./checkpoints/pretrain',
        metadata={"help": "The path for storing model."}
    )

@dataclass
class DataFinetuningArguments(DataTrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for pre-training.
    """
    epochs: int = field(
        default=500,
        metadata={"help": "The number of epoch."},
    )
    batch_size: int = field(
        default=512,
        metadata={"help": "The batch size."},
    )
    weight_decay: float = field(
        default=0.05,
        metadata={"help": "The weight decay in AdamW."},
    )
    learning_rate: float = field(
        default=5e-4,
        metadata={"help": "The learning rate."},
    )
    num_classes: Optional[int] = field(
        default=20,
        metadata={"help": "The number of classes in each dataset."},
    )
    pretrain: bool = field(
        default=False,
        metadata={"help": "Whether to pretrain or fintune."}
    )
    T_max: int = field(
        default=10,
        metadata={"help": "The t-max value for CosineAnnealingLR."},
    )
    model_save_path: str = field(
        default='./checkpoints/finetune/',
        metadata={"help": "The path for storing finetuning model."}
    )
    checkpoint_path: str = field(
        default='./checkpoints/epoch=489-step=137200.ckpt',
        metadata={"help": "The path for storing pretraining model."}
    )


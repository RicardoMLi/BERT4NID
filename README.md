# BERT4NID: An Intra and Inter Packet Representation with Pre-training Transformers for IoT Network Intrusion Detection

## Overview
This paper proposes a new IoT network intrusion detection model called Bidirectional Encoder Representations from Transformer for Network Intrusion Detection (BERT4NID). BERT4NID learns network traffic features from large-scale unlabeled raw traffic using proxy tasks during pre-training. For downstream tasks, it only requires a small amount of labeled data for fine-tuning, enabling efficient detection of various network attacks. Unlike traditional approaches that rely on handcrafted statistical features, BERT4NID directly utilizes raw network traffic as input, thereby reducing dependence on manual feature engineering and improving adaptability to diverse traffic environments. The proposed framework primarily learns representations from partially observable packet bytes, headers, and early flow-level correlations. Consequently, its effectiveness depends on whether meaningful traffic patterns remain observable within the network traffic. Specifically, we propose the Packet2Embedding technique to convert each packet in a bidirectional flow into token embedding. We also introduce two novel proxy tasks: Masked Byte Prediction (MBP), which helps the model learn semantic and structural packet information, and Same Flow Prediction (SFP), which teaches the model to understand contextual relationships between packets. Finally, we fine-tune the model using a small amount of labeled data for effective intrusion detection.

![BERT4NID's framework](https://github.com/RicardoMLi/BERT4NID/blob/main/images/Overview_of_BERT4NID.png)

## Requirements
* Python >= 3.7
* CUDA: 11.4
* GPU: NVIDIA GeForce RTX 4060Ti
* torch >= 1.13
* six >= 1.15.0
* scapy == 2.4.5
* numpy == 1.21.6
* shutil, random, json, pickle, binascii, flowcontainer, matplotlib
* argparse
* packaging
* tshark
* [UER-py](https://github.com/dbiir/UER-py)
* [SplitCap](https://www.netresec.com/?page=SplitCap)
* [scikit-learn](https://scikit-learn.org/stable/)
* For the mixed precision training you will need apex from NVIDIA
* For the pre-trained model conversion (related with TensorFlow) you will need TensorFlow
* For the tokenization with wordpiece model you will need [WordPiece](https://github.com/huggingface/tokenizers)

## Quick start
### Pre-process
To obtain the pre-training and fine-tuning network traffic data, follow the following steps:
 1. The original PCAP file needs to be processed using the [SplitCap](https://www.netresec.com/?page=SplitCap) tool to divide the raw network traffic into individual flow-level PCAP files, where each file corresponds to a single bidirectional flow.
 2. Run `vocab/main.py` to generate the network traffic corpus and vocab. Note you should change the `root_dir` path according to your settings.
 3. Run `preprocess/pretrain_dataset.py` to get the pre-training dataset.
    ```
    python preprocess/pretrain_dataset.py --corpus_path vocab/corpora.txt \
                                          --vocab_path vocab/vocab.txt \
                                          --dataset_path datasets/pretrained_bert_dataset.pt \
                                          --processes_num 8 --target bert
    ```
 4. Run 'preprocess/finetune_dataset.py' to get the fine-tuning dataset.
    ```
    python preprocess/finetune_dataset.py --dataset ustc \
                                          --split_session_folder /data/ustc/
                                          --level flow --output_path datasets/finetune_dataset.tsv
    ```

### Pre-training
To pre-train the BERT4NID model, please use `pre_training/pretrain.py` to start the pre-training process.
```
python pre_training/pretrain.py --dataset_path datasets/pretrained_bert_dataset.pt \
                                --vocab_path vocab/vocab.txt --output_model_path checkpoints/pretrained_bert.bin \
                                --total_steps 500000 --save_checkpoint_steps 10000 --batch_size 64 \
                                --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert
```

### Fine-tuning
To fine-tune the BERT4NID model, please use `fine_tuning/finetune.py` to start the fine-tuning process.
```
python fine_tuning/finetune.py --pretrained_model_path checkpoints/pretrained_bert.bin \
                               --vocab_path vocab/vocab.txt --train_path datasets/finetune_dataset_ustc_train.tsv \
                               --val_path datasets/finetune_dataset_ustc_val.tsv \
                               --test_path datasets/finetune_dataset_ustc_test.tsv \
                               --epochs_num 20 --batch_size 128 --embedding word_pos_seg --encoder transformer \
                               --mask fully_visible --seq_length 128 --learning_rate 2e-5 --output_model_path checkpoints/finetune_bert.bin
```

### Few-shot learning
To evaluate the few-shot learning capability of the BERT4NID model, please use `few_shot/few_shot.py` to start the few-shot learning process. Note you should change the parameter `n_shots` to get results of different few-shot settings.
```
python few_shot/few_shot.py --pretrained_model_path checkpoints/pretrained_bert.bin --vocab_path vocab/vocab.txt \
                            --train_path datasets/finetune_dataset_ustc_train.tsv --val_path datasets/finetune_dataset_ustc_val.tsv \
                            --test_path datasets/finetune_dataset_ustc_test.tsv --epochs_num 10 --batch_size 128 \
                            --embedding word_pos_seg --encoder transformer --mask fully_visible --seq_length 128 \
                            --learning_rate 5e-6 --output_model_path checkpoints/finetune_bert.bin --n_shots 0.01 

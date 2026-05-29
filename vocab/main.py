import binascii
import json
import os
import sys
import scapy.all as scapy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from uer.utils.seed import set_seed
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.process_packet import transform_packet
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

word_name = "./vocab/corpora.txt"
vocab_name = "./vocab/vocab.txt"


def cut(obj, sec):
    result = [obj[i:i + sec] for i in range(0, len(obj), sec)]
    remanent_count = len(result[0]) % 4
    if remanent_count == 0:
        pass
    else:
        result = [obj[i:i + sec + remanent_count] for i in range(0, len(obj), sec + remanent_count)]
    return result


def bigram_generation(packet_string, packet_len=256):
    result = ''
    sentence = cut(packet_string, 1)
    token_count = 0
    for sub_string_index in range(len(sentence)):
        if sub_string_index != (len(sentence) - 1):
            token_count += 1
            if token_count > packet_len:
                break
            else:
                merge_word_bigram = sentence[sub_string_index] + sentence[sub_string_index + 1]
        else:
            break
        result += merge_word_bigram
        result += ' '

    return result


def onegram_generation(packet_string, packet_len=256):
    sentence = [packet_string[i:i+2] for i in range(0, len(packet_string), 2)]
    result = ''
    token_count = 0

    for idx, sub_string in enumerate(sentence):
        if idx != len(sentence) - 1:
            token_count += 1
            if token_count > packet_len:
                break
            else:
                result += sub_string
                result += ' '
        else:
            break

    return result


def preprocess(pcap_dir):
    # pcap_dir: Zeus和Tinba这一层，即session文件的上一层文件夹
    print("now pre-process pcap_dir is %s" % pcap_dir)
    n = 0

    for parent, dirs, files in os.walk(pcap_dir):
        for file in files:
            n += 1
            pcap_name = os.path.join(parent, file)
            print("No.%d pacp is processed ... %s ..." % (n, file))
            packets = scapy.rdpcap(pcap_name)
            if len(packets) < 2:
                print(f'{file} contains less than 2 packets')
                continue

            words_txt = []
            bigram_string = ''
            valid_packet_num = 0
            for packet_index in range(len(packets)):
                word_packet = transform_packet(packets[packet_index].copy())
                if word_packet is None:
                    print(f"we just deal with ip packets.")
                    continue

                valid_packet_num += 1
                if valid_packet_num > 6:
                    print(f'we just get first 6 valid packets')
                    break

                words_string = binascii.hexlify(bytes(word_packet)).decode()
                bigram_string += bigram_generation(words_string)
                bigram_string += '\n'

            if valid_packet_num < 2:
                print(f'{file} contains less than 2 valid packets')
                continue
            else:
                words_txt.append(bigram_string + '\n')

            with open(word_name, 'a') as result_file:
                for words in words_txt:
                    result_file.write(words)

    print("finish preprocessing %d pcaps" % n)


def build_WP():
    # generate source dictionary,0-65535
    num_count = 256
    i = 0
    source_dictionary = {}
    # 'PAD':0,'UNK':1,'CLS':2,'SEP':3,'MASK':4
    while i < num_count:
        temp_string = '{:02x}'.format(i)
        source_dictionary[temp_string] = i
        i += 1
    # Initialize a tokenizer
    tokenizer = Tokenizer(models.WordPiece(vocab=source_dictionary, unk_token="[UNK]", max_input_chars_per_word=2))

    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.decoder = decoders.WordPiece()
    tokenizer.post_processor = processors.BertProcessing(sep=("[SEP]", 1), cls=('[CLS]', 2))

    # And then train
    trainer = trainers.WordPieceTrainer(vocab_size=261, min_frequency=2)
    tokenizer.train([word_name, word_name], trainer=trainer)

    # And Save it
    tokenizer.save("wordpiece.tokenizer.json", pretty=True)
    return 0


def build_vocab():
    json_file = open("wordpiece.tokenizer.json", 'r')
    json_content = json_file.read()
    json_file.close()
    vocab_json = json.loads(json_content)
    vocab_txt = ["[PAD]", "[SEP]", "[CLS]", "[UNK]", "[MASK]"]
    for item in vocab_json['model']['vocab']:
        vocab_txt.append(item)  # append key of vocab_json
    with open(vocab_name, 'w') as f:
        for word in vocab_txt:
            f.write(word + "\n")
    return 0


if __name__ == '__main__':

    set_seed(seed=3407)
    # 预训练数据集文件夹
    root_dir = r'E:\projects\datasets\test_dataset'
    label_names = os.listdir(root_dir)
    all_task = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for label_name in label_names:
            all_task.append(executor.submit(preprocess, os.path.join(root_dir, label_name)))

        for future in as_completed(all_task):
            future.result()

    # build vocab
    build_WP()
    build_vocab()

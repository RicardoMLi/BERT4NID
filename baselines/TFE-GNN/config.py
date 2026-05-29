'''
Training Configuration
'''
class Config:
    BATCH_SIZE = 102
    GRADIENT_ACCUMULATION = 5
    MAX_EPOCH = 20
    LR = 1e-2
    LR_MIN = 1e-4
    LABEL_SMOOTHING = 0
    WEIGHT_DECAY = 0
    WARM_UP = 0.1
    SEED = 32
    DROPOUT = 0.2
    DOWNSTREAM_DROPOUT = 0.0

    EMBEDDING_SIZE = 64
    H_FEATS = 128
    NUM_CLASSES = 14

    PMI_WINDOW_SIZE = 5
    PAD_TRUNC_DIGIT = 256
    FLOW_PAD_TRUNC_LENGTH = 50      # 每个flow或者traffic segment中最大的packet数量
    BYTE_PAD_TRUNC_LENGTH = 150     # 每个packet中payload length最大为150个字节
    HEADER_BYTE_PAD_TRUNC_LENGTH = 40     # 每个packet中header length最大为40字节
    ANOMALOUS_FLOW_THRESHOLD = 10000


'''
ISCX-VPN Dataset Configuration
'''
class ISCXVPNConfig(Config):
    TRAIN_DATA = r'/data1/zhz/ISCX-VPN-NonVPN-2016/VPN_SPLIT/TCP/train.npz'
    HEADER_TRAIN_DATA = r'/data1/zhz/ISCX-VPN-NonVPN-2016/VPN_SPLIT/TCP/header_train.npz'
    TEST_DATA = r'/data1/zhz/ISCX-VPN-NonVPN-2016/VPN_SPLIT/TCP/test.npz'
    HEADER_TEST_DATA = r'/data1/zhz/ISCX-VPN-NonVPN-2016/VPN_SPLIT/TCP/header_test.npz'

    TRAIN_GRAPH_DATA = r'/data1/zhz/ISCX-VPN-NonVPN-2016/VPN_SPLIT/TCP/train_graph.dgl'
    HEADER_TRAIN_GRAPH_DATA = r'/data1/zhz/ISCX-VPN-NonVPN-2016/VPN_SPLIT/TCP/header_train_graph.dgl'
    TEST_GRAPH_DATA = r'/data1/zhz/ISCX-VPN-NonVPN-2016/VPN_SPLIT/TCP/test_graph.dgl'
    HEADER_TEST_GRAPH_DATA = r'/data1/zhz/ISCX-VPN-NonVPN-2016/VPN_SPLIT/TCP/header_test_graph.dgl'

    MIX_MODEL_CHECKPOINT = r'./checkpoints/mix_model_iscx_vpn.pth'

    NUM_CLASSES = 6
    MAX_SEG_PER_CLASS = 9999
    NUM_WORKERS = 5

    BATCH_SIZE = 32
    GRADIENT_ACCUMULATION = 1
    MAX_EPOCH = 20
    LR = 1e-2
    LR_MIN = 1e-4
    LABEL_SMOOTHING = 0
    WEIGHT_DECAY = 0
    WARM_UP = 0.1
    SEED = 32
    DROPOUT = 0.2
    DOWNSTREAM_DROPOUT = 0.0
    EMBEDDING_SIZE = 64
    H_FEATS = 128

    DIR_PATH_DICT = {0: r'/data1/zhz/ISCX-VPN-NonVPN-2016/VPN_SPLIT/TCP/Chat',
                     1: r'/data1/zhz/ISCX-VPN-NonVPN-2016/VPN_SPLIT/TCP/Email',
                     2: r'/data1/zhz/ISCX-VPN-NonVPN-2016/VPN_SPLIT/TCP/File',
                     3: r'/data1/zhz/ISCX-VPN-NonVPN-2016/VPN_SPLIT/TCP/P2P',
                     4: r'/data1/zhz/ISCX-VPN-NonVPN-2016/VPN_SPLIT/TCP/Streaming',
                     5: r'/data1/zhz/ISCX-VPN-NonVPN-2016/VPN_SPLIT/TCP/VoIP',
                     }


'''
ISCX-NonVPN Dataset Configuration
'''
class ISCXNonVPNConfig(Config):
    TRAIN_DATA = r'/data1/zhz/ISCX-VPN-NonVPN-2016/NonVPN_SPLIT/TCP/train.npz'
    HEADER_TRAIN_DATA = r'/data1/zhz/ISCX-VPN-NonVPN-2016/NonVPN_SPLIT/TCP/header_train.npz'
    TEST_DATA = r'/data1/zhz/ISCX-VPN-NonVPN-2016/NonVPN_SPLIT/TCP/test.npz'
    HEADER_TEST_DATA = r'/data1/zhz/ISCX-VPN-NonVPN-2016/NonVPN_SPLIT/TCP/header_test.npz'

    TRAIN_GRAPH_DATA = r'/data1/zhz/ISCX-VPN-NonVPN-2016/NonVPN_SPLIT/TCP/train_graph.dgl'
    HEADER_TRAIN_GRAPH_DATA = r'/data1/zhz/ISCX-VPN-NonVPN-2016/NonVPN_SPLIT/TCP/header_train_graph.dgl'
    TEST_GRAPH_DATA = r'/data1/zhz/ISCX-VPN-NonVPN-2016/NonVPN_SPLIT/TCP/test_graph.dgl'
    HEADER_TEST_GRAPH_DATA = r'/data1/zhz/ISCX-VPN-NonVPN-2016/NonVPN_SPLIT/TCP/header_test_graph.dgl'

    MIX_MODEL_CHECKPOINT = r'./checkpoints/mix_model_iscx_nonvpn.pth'

    NUM_CLASSES = 6
    MAX_SEG_PER_CLASS = 9999
    NUM_WORKERS = 5

    BATCH_SIZE = 102
    GRADIENT_ACCUMULATION = 5
    MAX_EPOCH = 120
    LR = 1e-2
    LR_MIN = 1e-5
    LABEL_SMOOTHING = 0.01
    WEIGHT_DECAY = 0
    WARM_UP = 0.1
    SEED = 32
    DROPOUT = 0.1
    DOWNSTREAM_DROPOUT = 0.15
    EMBEDDING_SIZE = 64
    H_FEATS = 128

    DIR_PATH_DICT = {0: r'/data1/zhz/ISCX-VPN-NonVPN-2016/NonVPN_SPLIT/TCP/Chat',
                     1: r'/data1/zhz/ISCX-VPN-NonVPN-2016/NonVPN_SPLIT/TCP/Email',
                     2: r'/data1/zhz/ISCX-VPN-NonVPN-2016/NonVPN_SPLIT/TCP/File',
                     3: r'/data1/zhz/ISCX-VPN-NonVPN-2016/NonVPN_SPLIT/TCP/Streaming',
                     4: r'/data1/zhz/ISCX-VPN-NonVPN-2016/NonVPN_SPLIT/TCP/Video',
                     5: r'/data1/zhz/ISCX-VPN-NonVPN-2016/NonVPN_SPLIT/TCP/VoIP',
                     }


'''
ISCX-Tor Dataset Configuration
'''
class ISCXTorConfig(Config):
    TRAIN_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/train.npz'
    HEADER_TRAIN_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/header_train.npz'
    TEST_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/test.npz'
    HEADER_TEST_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/header_test.npz'

    TRAIN_GRAPH_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/train_graph.dgl'
    HEADER_TRAIN_GRAPH_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/header_train_graph.dgl'
    TEST_GRAPH_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/test_graph.dgl'
    HEADER_TEST_GRAPH_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/header_test_graph.dgl'

    MIX_MODEL_CHECKPOINT = r'./checkpoints/mix_model_iscx_tor.pth'

    NUM_CLASSES = 8
    MAX_SEG_PER_CLASS = 9999
    NUM_WORKERS = 5

    BATCH_SIZE = 32
    GRADIENT_ACCUMULATION = 1
    MAX_EPOCH = 100
    LR = 1e-2
    LR_MIN = 1e-4
    LABEL_SMOOTHING = 0
    WEIGHT_DECAY = 0
    WARM_UP = 0.1
    SEED = 32
    DROPOUT = 0.0
    DOWNSTREAM_DROPOUT = 0.0
    EMBEDDING_SIZE = 64
    H_FEATS = 128

    DIR_PATH_DICT = {0: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/Audio-Streaming',
                     1: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/Browsing',
                     2: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/Chat',
                     3: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/File',
                     4: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/Mail',
                     5: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/P2P',
                     6: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/Video-Streaming',
                     7: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/TOR_SPLIT/TCP/VoIP'
                     }


'''
ISCX-NonTor Dataset Configuration
'''
class ISCXNonTorConfig(Config):
    TRAIN_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/train.npz'
    HEADER_TRAIN_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/header_train.npz'
    TEST_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/test.npz'
    HEADER_TEST_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/header_test.npz'

    TRAIN_GRAPH_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/train_graph.dgl'
    HEADER_TRAIN_GRAPH_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/header_train_graph.dgl'
    TEST_GRAPH_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/test_graph.dgl'
    HEADER_TEST_GRAPH_DATA = r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/header_test_graph.dgl'

    MIX_MODEL_CHECKPOINT = r'./checkpoints/mix_model_iscx_nontor.pth'

    NUM_CLASSES = 8
    MAX_SEG_PER_CLASS = 9999
    NUM_WORKERS = 5

    BATCH_SIZE = 102
    GRADIENT_ACCUMULATION = 5
    MAX_EPOCH = 120
    LR = 1e-2
    LR_MIN = 1e-4
    LABEL_SMOOTHING = 0
    WEIGHT_DECAY = 0
    WARM_UP = 0.1
    SEED = 32
    DROPOUT = 0.05
    DOWNSTREAM_DROPOUT = 0.1
    EMBEDDING_SIZE = 64
    H_FEATS = 128

    DIR_PATH_DICT = {0: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/Audio',
                     1: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/Browsing',
                     2: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/Chat',
                     3: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/Email',
                     4: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/FTP',
                     5: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/P2P',
                     6: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/Video',
                     7: r'/data1/zhz/ISCX-Tor-NonTor-2017/Tor/Pcaps/NonTOR_SPLIT/TCP/VoIP',
                     }


class USTCConfig(Config):
    TRAIN_DATA = r'./data/ustc/train.npz'
    HEADER_TRAIN_DATA = r'./data/ustc/header_train.npz'
    TEST_DATA = r'./data/ustc/test.npz'
    HEADER_TEST_DATA = r'./data/ustc/header_test.npz'

    TRAIN_GRAPH_DATA = r'./data/ustc/train_graph.dgl'
    HEADER_TRAIN_GRAPH_DATA = r'./data/ustc/header_train_graph.dgl'
    TEST_GRAPH_DATA = r'./data/ustc/test_graph.dgl'
    HEADER_TEST_GRAPH_DATA = r'./data/ustc/header_test_graph.dgl'

    MIX_MODEL_CHECKPOINT = r'./checkpoints/mix_model_ustc.pth'

    NUM_CLASSES = 20
    MAX_SEG_PER_CLASS = 7000
    NUM_WORKERS = 16

    BATCH_SIZE = 256
    GRADIENT_ACCUMULATION = 1
    MAX_EPOCH = 20
    LR = 1e-2
    LR_MIN = 1e-4
    LABEL_SMOOTHING = 0
    WEIGHT_DECAY = 0
    WARM_UP = 0.1
    SEED = 32
    DROPOUT = 0.2
    DOWNSTREAM_DROPOUT = 0.0
    EMBEDDING_SIZE = 64
    H_FEATS = 128

    SEP_NPZ_FILE = {
        0: r'./data/ustc/BitTorrent/',
        1: r'./data/ustc/Facetime/',
        2: r'./data/ustc/FTP/',
        3: r'./data/ustc/Gmail/',
        4: r'./data/ustc/MySQL/',
        5: r'./data/ustc/Outlook/',
        6: r'./data/ustc/Skype/',
        7: r'./data/ustc/SMB/',
        8: r'./data/ustc/Weibo/',
        9: r'./data/ustc/WorldOfWarcraft/',
        10: r'./data/ustc/Cridex/',
        11: r'./data/ustc/Geodo/',
        12: r'./data/ustc/Htbot/',
        13: r'./data/ustc/Miuref/',
        14: r'./data/ustc/Neris/',
        15: r'./data/ustc/Nsis-ay/',
        16: r'./data/ustc/Shifu/',
        17: r'./data/ustc/Tinba/',
        18: r'./data/ustc/Virut/',
        19: r'./data/ustc/Zeus/',
    }

    DIR_PATH_DICT = {0: r'/data3/wzy/hyy/lzy/USTC-processed-2/split_sessions/BitTorrent.pcap',
                     1: r'/data3/wzy/hyy/lzy/USTC-processed-2/split_sessions/Facetime.pcap',
                     2: r'/data3/wzy/hyy/lzy/USTC-processed-2/split_sessions/FTP.pcap',
                     3: r'/data3/wzy/hyy/lzy/USTC-processed-2/split_sessions/Gmail.pcap',
                     4: r'/data3/wzy/hyy/lzy/USTC-processed-2/split_sessions/MySQL.pcap',
                     5: r'/data3/wzy/hyy/lzy/USTC-processed-2/split_sessions/Outlook.pcap',
                     6: r'/data3/wzy/hyy/lzy/USTC-processed-2/split_sessions/Skype.pcap',
                     7: r'/data3/wzy/hyy/lzy/USTC-processed-2/split_sessions/SMB.pcap',
                     8: r'/data3/wzy/hyy/lzy/USTC-processed-2/split_sessions/Weibo.pcap',
                     9: r'/data3/wzy/hyy/lzy/USTC-processed-2/split_sessions/WorldOfWarcraft.pcap',
                     10: r'/data3/wzy/hyy/lzy/USTC-processed-2/split_sessions/Cridex.pcap',
                     11: r'/data3/wzy/hyy/lzy/USTC-processed-2/split_sessions/Geodo.pcap',
                     12: r'/data3/wzy/hyy/lzy/USTC-processed-2/split_sessions/Htbot.pcap',
                     13: r'/data3/wzy/hyy/lzy/USTC-processed-2/split_sessions/Miuref.pcap',
                     14: r'/data3/wzy/hyy/lzy/USTC-processed-2/split_sessions/Neris.pcap',
                     15: r'/data3/wzy/hyy/lzy/USTC-processed-2/split_sessions/Nsis-ay.pcap',
                     16: r'/data3/wzy/hyy/lzy/USTC-processed-2/split_sessions/Shifu.pcap',
                     17: r'/data3/wzy/hyy/lzy/USTC-processed-2/split_sessions/Tinba.pcap',
                     18: r'/data3/wzy/hyy/lzy/USTC-processed-2/split_sessions/Virut.pcap',
                     19: r'/data3/wzy/hyy/lzy/USTC-processed-2/split_sessions/Zeus.pcap',
                     }


class MedConfig(Config):
    TRAIN_DATA = r'./data/med/train.npz'
    HEADER_TRAIN_DATA = r'./data/med/header_train.npz'
    TEST_DATA = r'./data/med/test.npz'
    HEADER_TEST_DATA = r'./data/med/header_test.npz'

    TRAIN_GRAPH_DATA = r'./data/med/train_graph.dgl'
    HEADER_TRAIN_GRAPH_DATA = r'./data/med/header_train_graph.dgl'
    TEST_GRAPH_DATA = r'./data/med/test_graph.dgl'
    HEADER_TEST_GRAPH_DATA = r'./data/med/header_test_graph.dgl'

    MIX_MODEL_CHECKPOINT = r'./checkpoints/mix_model_med.pth'

    NUM_CLASSES = 4
    MAX_SEG_PER_CLASS = 7000
    NUM_WORKERS = 6

    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION = 1
    MAX_EPOCH = 5
    LR = 1e-2
    LR_MIN = 1e-4
    LABEL_SMOOTHING = 0
    WEIGHT_DECAY = 0
    WARM_UP = 0.1
    SEED = 32
    DROPOUT = 0.2
    DOWNSTREAM_DROPOUT = 0.0
    EMBEDDING_SIZE = 64
    H_FEATS = 128

    SEP_NPZ_FILE = {
        0: r'./data/med/bashlite/',
        1: r'./data/med/mirai/',
        2: r'./data/med/normal/',
        3: r'./data/med/torii/',
    }

    DIR_PATH_DICT = {r'/data3/wzy/hyy/lzy/MedBIoT-processed/bashlite_mal_CC_all.pcap': 0,
                     r'/data3/wzy/hyy/lzy/MedBIoT-processed/bashlite_mal_spread_all.pcap': 0,
                     r'/data3/wzy/hyy/lzy/MedBIoT-processed/mirai_mal_CC_all.pcap': 1,
                     r'/data3/wzy/hyy/lzy/MedBIoT-processed/mirai_mal_spread_all.pcap': 1,
                     r'/data3/wzy/hyy/lzy/MedBIoT-processed/bashlite_leg.pcap': 2,
                     r'/data3/wzy/hyy/lzy/MedBIoT-processed/mirai_leg.pcap': 2,
                     r'/data3/wzy/hyy/lzy/MedBIoT-processed/torii_leg.pcap': 2,
                     r'/data3/wzy/hyy/lzy/MedBIoT-processed/torii_mal_all.pcap': 3,
                     }


class MQTTConfig(Config):
    TRAIN_DATA = r'./data/mqtt/train.npz'
    HEADER_TRAIN_DATA = r'./data/mqtt/header_train.npz'
    TEST_DATA = r'./data/mqtt/test.npz'
    HEADER_TEST_DATA = r'./data/mqtt/header_test.npz'

    TRAIN_GRAPH_DATA = r'./data/mqtt/train_graph.dgl'
    HEADER_TRAIN_GRAPH_DATA = r'./data/mqtt/header_train_graph.dgl'
    TEST_GRAPH_DATA = r'./data/mqtt/test_graph.dgl'
    HEADER_TEST_GRAPH_DATA = r'./data/mqtt/header_test_graph.dgl'

    MIX_MODEL_CHECKPOINT = r'./checkpoints/mix_model_mqtt.pth'

    NUM_CLASSES = 5
    MAX_SEG_PER_CLASS = 7000
    NUM_WORKERS = 12

    BATCH_SIZE = 256
    GRADIENT_ACCUMULATION = 1
    MAX_EPOCH = 20
    LR = 1e-2
    LR_MIN = 1e-4
    LABEL_SMOOTHING = 0
    WEIGHT_DECAY = 0
    WARM_UP = 0.1
    SEED = 32
    DROPOUT = 0.2
    DOWNSTREAM_DROPOUT = 0.0
    EMBEDDING_SIZE = 64
    H_FEATS = 128
    SEP_NPZ_FILE = {
        0: r'./data/mqtt/mqtt_bruteforce/',
        1: r'./data/mqtt/normal/',
        2: r'./data/mqtt/scan_A/',
        3: r'./data/mqtt/scan_sU/',
        4: r'./data/mqtt/sparta/'
    }

    DIR_PATH_DICT = {0: r'/data3/wzy/hyy/lzy/MQTT_IoT-2/split_sessions/mqtt_bruteforce.pcap',
                     1: r'/data3/wzy/hyy/lzy/MQTT_IoT-2/split_sessions/normal.pcap',
                     2: r'/data3/wzy/hyy/lzy/MQTT_IoT-2/split_sessions/scan_A.pcap',
                     3: r'/data3/wzy/hyy/lzy/MQTT_IoT-2/split_sessions/scan_sU.pcap',
                     4: r'/data3/wzy/hyy/lzy/MQTT_IoT-2/split_sessions/sparta.pcap'
                     }


class KitsuneConfig(Config):
    TRAIN_DATA = r'./data/kitsune/train.npz'
    HEADER_TRAIN_DATA = r'./data/kitsune/header_train.npz'
    TEST_DATA = r'./data/kitsune/test.npz'
    HEADER_TEST_DATA = r'./data/kitsune/header_test.npz'

    TRAIN_GRAPH_DATA = r'./data/kitsune/train_graph.dgl'
    HEADER_TRAIN_GRAPH_DATA = r'./data/kitsune/header_train_graph.dgl'
    TEST_GRAPH_DATA = r'./data/kitsune/test_graph.dgl'
    HEADER_TEST_GRAPH_DATA = r'./data/kitsune/header_test_graph.dgl'

    MIX_MODEL_CHECKPOINT = r'./checkpoints/mix_model_kitsune.pth'

    NUM_CLASSES = 9
    MAX_SEG_PER_CLASS = 7000
    NUM_WORKERS = 12

    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION = 1
    MAX_EPOCH = 20
    LR = 1e-2
    LR_MIN = 1e-4
    LABEL_SMOOTHING = 0
    WEIGHT_DECAY = 0
    WARM_UP = 0.1
    SEED = 32
    DROPOUT = 0.2
    DOWNSTREAM_DROPOUT = 0.0
    EMBEDDING_SIZE = 64
    H_FEATS = 128

    SEP_NPZ_FILE = {
        0: r'./data/kitsune/Active_Wiretap/',
        1: r'./data/kitsune/ARP_MitM/',
        2: r'./data/kitsune/Fuzzing/',
        3: r'./data/kitsune/Mirai/',
        4: r'./data/kitsune/OS_Scan/',
        5: r'./data/kitsune/SSDP_Flood/',
        6: r'./data/kitsune/SSL_Renegotiation/',
        7: r'./data/kitsune/SYN_DoS/',
        8: r'./data/kitsune/Video_Injection/'
    }

    DIR_PATH_DICT = {0: r'/data3/wzy/hyy/lzy/Kitsune-processed/split_sessions/Active_Wiretap.pcap',
                     1: r'/data3/wzy/hyy/lzy/Kitsune-processed/split_sessions/ARP_MitM.pcap',
                     2: r'/data3/wzy/hyy/lzy/Kitsune-processed/split_sessions/Fuzzing.pcap',
                     3: r'/data3/wzy/hyy/lzy/Kitsune-processed/split_sessions/Mirai.pcap',
                     4: r'/data3/wzy/hyy/lzy/Kitsune-processed/split_sessions/OS_Scan.pcap',
                     5: r'/data3/wzy/hyy/lzy/Kitsune-processed/split_sessions/SSDP_Flood.pcap',
                     6: r'/data3/wzy/hyy/lzy/Kitsune-processed/split_sessions/SSL_Renegotiation.pcap',
                     7: r'/data3/wzy/hyy/lzy/Kitsune-processed/split_sessions/SYN_DoS.pcap',
                     8: r'/data3/wzy/hyy/lzy/Kitsune-processed/split_sessions/Video_Injection.pcap'
                     }


class EdgeConfig(Config):
    TRAIN_DATA = r'./data/edge/train.npz'
    HEADER_TRAIN_DATA = r'./data/edge/header_train.npz'
    TEST_DATA = r'./data/edge/test.npz'
    HEADER_TEST_DATA = r'./data/edge/header_test.npz'

    TRAIN_GRAPH_DATA = r'./data/edge/train_graph.dgl'
    HEADER_TRAIN_GRAPH_DATA = r'./data/edge/header_train_graph.dgl'
    TEST_GRAPH_DATA = r'./data/edge/test_graph.dgl'
    HEADER_TEST_GRAPH_DATA = r'./data/edge/header_test_graph.dgl'

    MIX_MODEL_CHECKPOINT = r'./checkpoints/mix_model_edge.pth'

    NUM_CLASSES = 14
    MAX_SEG_PER_CLASS = 7000
    NUM_WORKERS = 6

    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION = 1
    MAX_EPOCH = 5
    LR = 1e-2
    LR_MIN = 1e-4
    LABEL_SMOOTHING = 0
    WEIGHT_DECAY = 0
    WARM_UP = 0.1
    SEED = 32
    DROPOUT = 0.2
    DOWNSTREAM_DROPOUT = 0.0
    EMBEDDING_SIZE = 64
    H_FEATS = 128

    SEP_NPZ_FILE = {
        0: r'./data/med/Backdoor_attack/',
        1: r'./data/med/DDoS_HTTP_Flood_attacks/',
        2: r'./data/med/DDoS_ICMP_Flood_attacks/',
        3: r'./data/med/DDoS_TCP_SYN_Flood_attacks/',
        4: r'./data/med/DDoS_UDP_Flood_attacks/',
        5: r'./data/med/MITM_Attack/',
        6: r'./data/med/OS_Fingerprinting_attack/',
        7: r'./data/med/Password_attacks/',
        8: r'./data/med/PortScanning_attack/',
        9: r'./data/med/Ransomware_attack/',
        10: r'./data/med/SQL_injection_attack/',
        11: r'./data/med/Uploading_attack/',
        12: r'./data/med/XSS_attacks/',
        13: r'./data/med/Normal/'
    }

    DIR_PATH_DICT = {r'/data3/wzy/hyy/lzy/Edge_IIoT-processed/Backdoor_attack.pcap': 0,
                     r'/data3/wzy/hyy/lzy/Edge_IIoT-processed/DDoS_HTTP_Flood_attacks.pcap': 1,
                     r'/data3/wzy/hyy/lzy/Edge_IIoT-processed/DDoS_ICMP_Flood_attacks.pcap': 2,
                     r'/data3/wzy/hyy/lzy/Edge_IIoT-processed/DDoS_TCP_SYN_Flood_attacks.pcap': 3,
                     r'/data3/wzy/hyy/lzy/Edge_IIoT-processed/DDoS_UDP_Flood_attacks.pcap': 4,
                     r'/data3/wzy/hyy/lzy/Edge_IIoT-processed/MITM_Attack.pcap': 5,
                     r'/data3/wzy/hyy/lzy/Edge_IIoT-processed/OS_Fingerprinting_attack.pcap': 6,
                     r'/data3/wzy/hyy/lzy/Edge_IIoT-processed/Password_attacks.pcap': 7,
                     r'/data3/wzy/hyy/lzy/Edge_IIoT-processed/PortScanning_attack.pcap': 8,
                     r'/data3/wzy/hyy/lzy/Edge_IIoT-processed/Ransomware_attack.pcap': 9,
                     r'/data3/wzy/hyy/lzy/Edge_IIoT-processed/SQL_injection_attack.pcap': 10,
                     r'/data3/wzy/hyy/lzy/Edge_IIoT-processed/Uploading_attack.pcap': 11,
                     r'/data3/wzy/hyy/lzy/Edge_IIoT-processed/XSS_attacks.pcap': 12,
                     r'/data3/wzy/hyy/lzy/Edge_IIoT-processed/Distance.pcap': 13,
                     r'/data3/wzy/hyy/lzy/Edge_IIoT-processed/Flame_Sensor.pcap': 13,
                     r'/data3/wzy/hyy/lzy/Edge_IIoT-processed/Heart_Rate.pcap': 13,
                     r'/data3/wzy/hyy/lzy/Edge_IIoT-processed/IR_Receiver.pcap': 13,
                     r'/data3/wzy/hyy/lzy/Edge_IIoT-processed/Modbus.pcap': 13,
                     r'/data3/wzy/hyy/lzy/Edge_IIoT-processed/phValue.pcap': 13,
                     r'/data3/wzy/hyy/lzy/Edge_IIoT-processed/Soil_Moisture.pcap': 13,
                     r'/data3/wzy/hyy/lzy/Edge_IIoT-processed/Sound_Sensor.pcap': 13,
                     r'/data3/wzy/hyy/lzy/Edge_IIoT-processed/Temperature_and_Humidity.pcap': 13,
                     r'/data3/wzy/hyy/lzy/Edge_IIoT-processed/Water_Level.pcap': 13
                     }

class CTUConfig(Config):
    # FLOW_PAD_TRUNC_LENGTH = 20    # 每个flow或者traffic segment中最大的packet数量
    NAME = 'CTU-13'

    TRAIN_DATA = r'./data/ctu/train.npz'
    HEADER_TRAIN_DATA = r'./data/ctu/header_train.npz'
    TEST_DATA = r'./data/ctu/test.npz'
    HEADER_TEST_DATA = r'./data/ctu/header_test.npz'

    TRAIN_GRAPH_DATA = r'./data/ctu/train_graph.dgl'
    HEADER_TRAIN_GRAPH_DATA = r'./data/ctu/header_train_graph.dgl'
    TEST_GRAPH_DATA = r'./data/ctu/test_graph.dgl'
    HEADER_TEST_GRAPH_DATA = r'./data/ctu/header_test_graph.dgl'

    MIX_MODEL_CHECKPOINT = r'./checkpoints/mix_model_ctu.pth'
    NUM_CLASSES = 8
    BATCH_SIZE = 38
    MAX_SEG_PER_CLASS = 9999
    NUM_WORKERS = 12
    GRADIENT_ACCUMULATION = 1
    MAX_EPOCH = 20
    LR = 1e-2
    LR_MIN = 1e-4
    LABEL_SMOOTHING = 0
    WEIGHT_DECAY = 0
    WARM_UP = 0.1
    SEED = 32
    DROPOUT = 0.2
    DOWNSTREAM_DROPOUT = 0.0
    EMBEDDING_SIZE = 64
    H_FEATS = 128

    SEP_NPZ_FILE = {
        0: r'/data/hyy/lzy/projects/TFE-GNN/data/ctu/DonBot',
        1: r'/data/hyy/lzy/projects/TFE-GNN/data/ctu/Murlo',
        2: r'/data/hyy/lzy/projects/TFE-GNN/data/ctu/Neris',
        3: r'/data/hyy/lzy/projects/TFE-GNN/data/ctu/NSIS.ay',
        4: r'/data/hyy/lzy/projects/TFE-GNN/data/ctu/Rbot',
        5: r'/data/hyy/lzy/projects/TFE-GNN/data/ctu/Sogou',
        6: r'/data/hyy/lzy/projects/TFE-GNN/data/ctu/Virut',
        7: r'/data/hyy/lzy/projects/TFE-GNN/data/ctu/Normal'
    }

    DIR_PATH_DICT = {
        0: r'/data/hyy/lzy/processed_data/ctu/split_sessions/DonBot',
        1: r'/data/hyy/lzy/processed_data/ctu/split_sessions/Murlo',
        2: r'/data/hyy/lzy/processed_data/ctu/split_sessions/Neris',
        3: r'/data/hyy/lzy/processed_data/ctu/split_sessions/NSIS.ay',
        4: r'/data/hyy/lzy/processed_data/ctu/split_sessions/Rbot',
        5: r'/data/hyy/lzy/processed_data/ctu/split_sessions/Sogou',
        6: r'/data/hyy/lzy/processed_data/ctu/split_sessions/Virut',
        7: r'/data/hyy/lzy/processed_data/ctu/split_sessions/Normal'
    }


if __name__ == '__main__':
    config = Config()

from transformers import BertTokenizer
import json
import torch
import time

# GENERAL
SEED = 123
EXP_NUMBER = 7
# DATA
MIN_VOTE_UP_COUNT = 100  # 100
MIN_CONTENT_LENGTH = 64
MAX_CONTENT_LENGTH = 2500
MIN_TAGS_NUM = 2
MAX_TAGS_NUM = 6
SOURCE_DATA_PATH = "./data/"
SOURCE_DATA_FILE = f"{SOURCE_DATA_PATH}multi_tags_data_exp_{EXP_NUMBER}.dat.json"
LEGAL_TAGS_PATH = f'./data/tags_exp_{EXP_NUMBER}.dat.json'
RAW = False  # 是否token数据，False则加载上次token好的cache

MAX_TAGS_LENGTH = 50

N_CTX = 1024
N_CTX1 = 1800

assert MAX_CONTENT_LENGTH - N_CTX >= 100

# TRAIN
DEVICE_IDs = '0'
DEVICE_TYPE = torch.device("cuda" if torch.cuda.is_available() and int(DEVICE_IDs) >= 0 else "cpu")

VOCAB_PATH = './data/vocab.txt'


def get_path():
    return './data/train_image/{}.png'.format(int(round(time.time() * 1000)))


def get_path_predict():
    return './data/image_predict/{}.png'.format(int(round(time.time() * 1000)))


TOKENIZER = BertTokenizer(vocab_file=VOCAB_PATH, do_lower_case=True)  # 初始化tokenizer
# 将[space]作为一个分割整体，例如："我爱[Space]中国。"，使用原始tokenizer分词结果为"['我', '爱', '[', 'Space', ']', '中', '国', '。']";
# 增加分割符号后的结果为"['我', '爱', '[Space]', '中', '国', '。']"
# TOKENIZER.add_tokens("[Space]", special_tokens=True)
TOKENIZER.add_special_tokens(
    {'additional_special_tokens': ["[IMG]", "[Space]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[Comment]", "[Content]"]})

CONTENT_TYPE_ID = TOKENIZER.convert_tokens_to_ids("[Content]")
TAGS_TYPE_ID = TOKENIZER.convert_tokens_to_ids("[Tags]")
UNK_ID = TOKENIZER.convert_tokens_to_ids("[UNK]")
SEP_ID = TOKENIZER.convert_tokens_to_ids("[SEP]")
COMMENT_ID = TOKENIZER.convert_tokens_to_ids("[Comment]")
IMG_ID = TOKENIZER.convert_tokens_to_ids("[IMG]")

DATA_CACHE_PATH = './data/cached_data.cache'  # 生成缓存数据的存放路径
MODEL_CONFIG_PATH = './data/config.json'
N_HEAD = 12
# N_HEAD = 12
N_LAYER = 10
# N_LAYER = 10
VOCAB_SIZE = len(TOKENIZER)
MULTI_GPU = False

# PRETRAINED_MODEL_PATH = './generation/tags/v2/output_model/checkpoint-2021-03-02-11-28-01/'
# PRETRAINED_MODEL_PATH = './generation/tags/v2/output_model/checkpoint-5-2021-04-09-11-29-21/'
# PRETRAINED_MODEL_PATH = './output_model/checkpoint-7-2021-05-25-13-53-07/'

PRETRAINED_MODEL_PATH = None  # '预训练的GPT2模型的路径', GPT2介于预训练直接做任务之间，是否需要预训练未知
EPOCHS = 50
TRAIN_BATCH_SIZE = 12
TEST_BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WARMUP_PROPORTION = 0.1  # 'warm up概率，即训练总步长的百分之多少，进行warm up'
ADAM_EPSILON = 1e-8  # 'Adam优化器的epsilon值'
LOGGING_STEPS = 20
EVAL_STEPS = 4000
GRADIENT_ACCUMULATION_STEPS = 4  # '梯度积累'
MAX_GRAD_NORM = 1.0
MODEL_OUTPUT_PATH = './output_model/'

REPETITION_PENALTY = 1.2  # 重复处罚率
TOP_K = 5  # 解码时保留概率最高的多少个标记
TOP_P = 0.95  # 解码时保留概率累加大于多少的标记

MODEL_PREDICT_PATH = "./output_model/checkpoint-1-2021-05-26-21-42-19/"
PREDICT_FILE = './data/real_test.json'
GEN_TIMES = 5
BATCH_SIZE = 1

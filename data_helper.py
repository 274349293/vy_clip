import torch
import random
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import config as hyper_args
from skimage import io
from pymongo import MongoClient
import time
import logging
from torchvision import transforms
from PIL import Image
from io import BytesIO
import requests
import numpy as np
import warnings

warnings.filterwarnings('error')

# 代理服务器
proxyHost = "http-dyn.abuyun.com"
proxyPort = "9020"

# 代理隧道验证信息
proxyUser = "H0Y97L3F9871C4RD"
proxyPass = "8A6443D94EEFBFB7"

proxyMeta = "http://%(user)s:%(pass)s@%(host)s:%(port)s" % {
    "host": proxyHost,
    "port": proxyPort,
    "user": proxyUser,
    "pass": proxyPass,
}

proxies = {
    "http": proxyMeta,
    "https": proxyMeta,
}


class DataPrepare:
    def __init__(self, logger):
        self.__logger = logger

        self.__data = self.load_data()
        self.tokenized_data = self.tokenize_data()

        random.seed(hyper_args.SEED)
        random.shuffle(self.tokenized_data)

    def load_data(self):
        self.__logger.info("Loading raw data")

        collection = \
            MongoClient('mongodb://' + 'admin' + ':' + "WS8uHk499Kjf%4s#kzn75G" + '@' + '172.16.8.9' + ':' + '27017',
                        authMechanism='SCRAM-SHA-1')['TheEye']['huxiu_article']
        data_raw = collection.find()
        data = self.filter_data(data_raw)

        self.__logger.info(f"Data size: {len(data)}")
        self.__logger.info("Loading raw data finish")
        self.__logger.info("Transform raw data")

        return data

    def filter_data(self, data_raw):
        data = []

        for x in data_raw:
            data_dict = {}

            if x['content'] == '':
                continue
            if len(x['content']) < hyper_args.MIN_CONTENT_LENGTH or len(
                    x['content']) > hyper_args.MAX_CONTENT_LENGTH:
                continue
            else:
                data_dict['content'] = x['content'].replace("[IMG at Here]", "[IMG]")
                data_dict['content'] = data_dict['content'].replace(" ", "[Space]")
            if len(x['content_pic_urls']) > 0:
                data_dict['url'] = x['content_pic_urls']
            else:
                data_dict['url'] = []
            if x['cover_pic_url']:
                data_dict['content'] = "[IMG]" + data_dict['content']
                data_dict['url'].insert(0, x['cover_pic_url'])

            try:
                assert data_dict['content'].count("[IMG]") == len(data_dict['url'])
            except Exception:
                Oj_id = x["_id"]
                self.__logger.info(f'[IMG] num != url num, data from mongo is {Oj_id}')
                continue

            if data_dict['url']:
                data.append(data_dict)
            if len(data) > 10:
                break

        return data

    @staticmethod
    def tokenize_one_sample(sample):
        """
        数据处理函数
        Args:
            sample: 一个字典，包含内容和评论，格式为{"content": content, "url": [url,url..]}

        Returns:

        """

        data_transform = {
            "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "val": transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

        input_ids = []
        token_type_ids = []

        url_list = sample['url']
        # 对内容进行tokenizer.tokenize分词
        content = sample["content"]

        # 预截断, 加速用
        content_size = hyper_args.N_CTX1
        content = content[:content_size]

        content_tokens = hyper_args.TOKENIZER.tokenize(content)
        # 对内容进行tokenizer.tokenize分词，注意tokenizer中已经将[Space]作为一个分隔符，不会切割成多个字符

        # 判断如果正文过长，进行截断
        content_tokens_size = hyper_args.N_CTX - 2
        content_tokens = content_tokens[:content_tokens_size]

        # 生成模型所需的input_ids和token_type_ids
        '''cls'''
        input_ids.append(hyper_args.TOKENIZER.cls_token_id)
        token_type_ids.append(hyper_args.CONTENT_TYPE_ID)

        '''content'''
        input_ids.extend(hyper_args.TOKENIZER.convert_tokens_to_ids(content_tokens))
        token_type_ids.extend([hyper_args.CONTENT_TYPE_ID] * len(content_tokens))

        '''sep'''
        input_ids.append(hyper_args.TOKENIZER.sep_token_id)
        token_type_ids.append(hyper_args.CONTENT_TYPE_ID)

        '''image'''
        image_index = [i for i, x in enumerate(input_ids) if x == hyper_args.IMG_ID]
        input_image = []
        for url in url_list:

            try:
                response = requests.get(url, proxies=proxies)
                img = np.asarray(Image.open(BytesIO(response.content)))
                path = hyper_args.get_path()
                io.imsave(path, img)
                image = data_transform['train'](Image.open(path).convert('RGB'))
                for index in image_index:
                    token_type_ids[index] = hyper_args.IMG_ID
                input_image.append(image)
            except Exception as e:
                return None

        # 判断input_ids与token_type_ids长度是否一致
        assert len(input_ids) == len(token_type_ids)
        # 判断input_ids长度是否小于等于最大长度
        assert len(input_ids) <= hyper_args.N_CTX

        return input_ids, token_type_ids, input_image, image_index

    def tokenize_data(self):

        self.__logger.info(f"All cutting size (tokens size): {hyper_args.N_CTX}")
        self.__logger.info("Tokenize data begin")

        if hyper_args.RAW:

            data_set = []
            total_image = 0
            for idx, sample in tqdm(enumerate(self.__data), desc='Tokenizing data and analysis image url'):
                sample_ids = self.tokenize_one_sample(sample)
                if sample_ids is None:
                    continue
                input_ids, token_type_ids, input_image, image_index = sample_ids
                total_image = total_image + len(image_index)
                data_set.append(
                    {"input_ids": input_ids, "token_type_ids": token_type_ids, "input_image": input_image,
                     "image_index": image_index})
            self.__logger.info("Tokenize data finish")
            self.__logger.info(f"total image num {total_image}")
            self.__logger.info(f"Cached data in: {hyper_args.DATA_CACHE_PATH}")
            torch.save({"data_set": data_set}, hyper_args.DATA_CACHE_PATH)

        else:
            self.__logger.info(f"Load data from cache: {hyper_args.DATA_CACHE_PATH}")
            data_set = torch.load(hyper_args.DATA_CACHE_PATH)["data_set"]
            self.__logger.info(f"Load data success")

        self.__logger.info(f"Tokenized data size: {len(data_set)}")

        return data_set


class MyDataSet(torch.utils.data.Dataset):
    """评论生成模型所需要的数据类"""

    def __init__(self, data_set):
        """
        初始化函数
        Args:
            data_set: refer to class DataPrepare.tokenized_data

        """
        self.data_set = data_set

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        instance = self.data_set[idx]
        return instance


def collate_func(batch_data):
    """
    DataLoader所需的collate_fun函数，将数据处理成tensor形式
    Args:
        batch_data: batch数据

    Returns:

    """

    batch_size = len(batch_data)

    # 如果batch_size为0，则返回一个空字典
    if batch_size == 0:
        return {}
    input_ids_list, token_type_ids_list, image_feature_list, image_index_list = [], [], [], []
    for instance in batch_data:
        # 按照batch中的最大数据长度,对数据进行padding填充
        input_ids_temp = instance["input_ids"]
        token_type_ids_temp = instance["token_type_ids"]
        image_feature_temp = instance['input_image']
        image_index_temp = instance['image_index']
        # 将input_ids_temp和token_type_ids_temp添加到对应的list中
        input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
        token_type_ids_list.append(torch.tensor(token_type_ids_temp, dtype=torch.long))
        image_feature_list.append(image_feature_temp)
        image_index_list.append(image_index_temp)
    # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
            "token_type_ids": pad_sequence(token_type_ids_list, batch_first=True, padding_value=0),
            "input_image": image_feature_list, "image_index": image_index_list}


def create_logger(task_name, log_root="../log/"):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        f'%(asctime)s - {task_name} - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


if __name__ == '__main__':

    _logger = create_logger('test-data')
    _ = DataPrepare(logger=_logger)
    for x in _.tokenized_data:
        print(x['token_type_ids'])
        exit()

import numpy as np
import torch
import json
from tqdm import tqdm
import random
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
import config as hyper_args
import warnings
from utils.data_io import mkdir, check_dir
from utils.char import remove_control_chr, remove_url, merge_space, remove_line_scape, remove_html_tags
from pymongo import MongoClient
from skimage import io
from torchvision import transforms
from PIL import Image
from io import BytesIO
import requests
from concurrent.futures import ThreadPoolExecutor

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


class LoadDataFromSource:
    def __init__(self, logger):
        self.logger = logger
        self.min_vote_up = hyper_args.MIN_VOTE_UP_COUNT
        self.exp_number = hyper_args.EXP_NUMBER
        self.data_path = hyper_args.SOURCE_DATA_PATH
        self.file = hyper_args.SOURCE_DATA_FILE
        self.clean_cache_data = hyper_args.CLEAN_CACHE_DATA

        db_uri = 'mongodb://' + 'admin' + ':' + "WS8uHk499Kjf%4s#kzn75G" + '@' + '172.16.8.9' + ':' + '27017'
        client = MongoClient(db_uri, authMechanism='SCRAM-SHA-1', connect=False)
        self.db = client["TheEye"]
        self.img_label1 = "*****"
        self.img_label2 = '[IMG at Here]'

    def content_format(self, content):
        content = remove_control_chr(content)
        content = remove_url(content)
        try:
            content = remove_html_tags(content)
        except Warning:
            return None
        content = merge_space(content)
        content = remove_line_scape(content)
        content = content.replace(self.img_label1, "")
        content = content.replace(self.img_label2, "[IMG]")
        return content

    def index_answer_data(self):
        self.logger.info("Indexing answers begin")

        answer_dict = {}
        answer_count = 0

        for item in tqdm(self.db['zhihu_answer'].find().limit(600000)):

            try:
                vote_up = int(item["vote_up_count"])
                content = item['content']
                question_id = str(item["parents_id"][0])
                image_url_list = item['content_pic_urls']

            except KeyError:
                continue

            # 小于一定赞数跳过
            if vote_up < self.min_vote_up:
                continue

            content = self.content_format(content)

            try:
                assert content.count("[IMG]") == len(image_url_list)
            except Exception:
                # Oj_id = item["_id"]
                # self.logger.warning(f'[IMG] num != url num, data from mongo is {Oj_id}')
                continue

            if not content:
                continue

            if question_id in answer_dict:
                answer_dict[question_id].append([content, image_url_list])
            else:
                answer_dict[question_id] = [[content, image_url_list]]

            answer_count += 1

        self.logger.info("finish")
        self.logger.info(f"answer count: {answer_count}")

        return answer_dict

    def index_question_data(self, question_dict, collection_name):
        question_count = 0
        for item in tqdm(self.db[collection_name].find()):
            try:
                tags = [x.lower() for x in item['tags']]
                title = item['title']
                intro = item['content']
                question_id = str(item['doc_id'])
            except KeyError:
                continue

            if question_id not in question_dict:
                question_dict[question_id] = [tags, title]
                question_count += 1
        return question_count

    def index_questions_data(self):
        question_dict = {}
        self.logger.info("Indexing questions begin")
        question_count = self.index_question_data(question_dict, collection_name='zhihu_question')
        self.logger.info("Indexing hot questions begin")
        question_count += self.index_question_data(question_dict, collection_name='zhihu_hot_question')
        self.logger.info("finish")
        self.logger.info(f"question count: {question_count}")

        return question_dict

    def combine_data(self, answer_dict, question_dict):
        self.logger.info("Combine data begin")
        data_list = []
        for question_id in tqdm(answer_dict):
            answers_content = answer_dict[question_id]
            if question_id not in question_dict:
                continue
            question_tags = ",".join(question_dict[question_id][0])
            question_title = question_dict[question_id][1]

            for answer in answers_content:
                answer_data = answer[0]
                image_url_list = answer[1]
                data_list.append([f"{question_tags}\t{answer_data}", image_url_list])
        self.logger.info("Finish")
        self.logger.info(f"Pair data num: {len(data_list)}")

        return data_list

    def load(self):
        self.logger.info("Load source data begin")

        self.logger.info(f"File will be saved in: {self.file}")

        if check_dir(self.file) and not self.clean_cache_data:
            self.logger.info(f"Already find data cache, if not you want, please remove")
            self.logger.info(f"Load data from cache")
            with open(self.file, 'r') as f:
                combined_data = json.load(f)
            self.logger.info(f"Load data success")
        else:
            self.logger.info(f"Load data from data base")
            answer_dict = self.index_answer_data()
            question_dict = self.index_questions_data()
            combined_data = self.combine_data(answer_dict, question_dict)
            mkdir(self.data_path)

            self.logger.info('Cache tags data begin')
            with open(self.file, 'w') as f:
                json.dump(combined_data, f, indent=4, ensure_ascii=False)
            self.logger.info('Cache tags data success')

        return combined_data


class DataPrepare:
    def __init__(self, logger, data):
        self.__logger = logger
        self.__logger.info(f"Raw mode: {hyper_args.CLEAN_TOKENIZE}")
        self.data = data

        self.__all_tags = set()
        self.__useful_tags = set()
        random.seed(hyper_args.SEED)

    def store_tags(self):
        self.__logger.info(f'All tags num: {len(self.__all_tags)}')
        self.__logger.info(f'Use tags num: {len(self.__useful_tags)}')
        self.__logger.info(f'Store use tags in path: {hyper_args.LEGAL_TAGS_PATH}')
        with open(hyper_args.LEGAL_TAGS_PATH, 'w') as f:
            json.dump(list(self.__useful_tags), f, indent=4, ensure_ascii=False)
        self.__logger.info('Store finish')

    def drop_blacklist_tags(self, tags, black_tags_num, mapping_tags_num):
        """

        :param mapping_tags_num:
        :param black_tags_num:
        :param tags: list
        :return: list
        """

        new_tags = []
        for tag in tags:
            self.__all_tags.add(tag)
            if tag in hyper_args.TAGS_BLACK_SET:
                black_tags_num += 1
                continue
            if tag in hyper_args.TAGS_MAPPING_DICT:
                mapping_tags_num += 1
                tag = hyper_args.TAGS_MAPPING_DICT[tag]
            new_tags.append(tag)
            self.__useful_tags.add(tag)
        return new_tags, black_tags_num, mapping_tags_num

    def filter_data(self, data):
        self.__logger.info(f"Accept min content length: {hyper_args.MIN_CONTENT_LENGTH}")
        self.__logger.info(f"Accept max content length: {hyper_args.MAX_CONTENT_LENGTH}")
        self.__logger.info(f"Accept min tags length: {hyper_args.MIN_TAGS_NUM}")
        self.__logger.info(f"Accept max tags length: {hyper_args.MAX_TAGS_NUM}")
        self.__logger.info(f'Black list tags num: {len(hyper_args.TAGS_BLACK_SET)}')
        self.__logger.info(f'Mapping tags kind num: {len(hyper_args.TAGS_MAPPING_DICT)}')

        clean_data = []
        black_tags_num = 0
        mapping_tags_num = 0
        tags_over_size_content_num = 0
        content_over_size_num = 0
        for sample in data:
            tags = sample[0].split('\t')[0]
            tag_list = tags.split(',')
            tag_list, black_tags_num, mapping_tags_num = self.drop_blacklist_tags(tag_list, black_tags_num,
                                                                                  mapping_tags_num)
            if len(tag_list) < hyper_args.MIN_TAGS_NUM or len(tag_list) > hyper_args.MAX_TAGS_NUM:
                tags_over_size_content_num += 1
                continue
            content = sample[0].split('\t')[1]
            if len(content) < hyper_args.MIN_CONTENT_LENGTH or len(content) > hyper_args.MAX_CONTENT_LENGTH:
                content_over_size_num += 1
                continue

            content = content.replace(" ", "[Space]")
            tag_list = ','.join(tag_list).replace(" ", "[Space]")
            new_sample = [f'{tag_list}\t{content}', sample[1]]

            clean_data.append(new_sample)
        self.__logger.info(f'Del over size tags content: {tags_over_size_content_num}')
        self.__logger.info(f'Del over size content: {content_over_size_num}')
        self.__logger.info(f'Del black list tags: {black_tags_num}')
        self.__logger.info(f'Mapping tags: {mapping_tags_num}')

        return clean_data

    def build_data(self, data):
        """
           整理数据格式
           Args:
           Returns:

           """

        data_set = []
        avg_content_length = 0
        content_length_array = []
        data_len = len(data)
        for sample in data:
            tags = sample[0].split('\t')[0]
            content = sample[0].split('\t')[1]
            image_url_list = sample[1]
            data_set.append({"content": content, "tags": tags, "image": image_url_list})

            avg_content_length += len(content) / data_len
            content_length_array.append(len(content))

        content_length_array.sort()
        mid_content_length = content_length_array[int(data_len / 2)]

        self.__logger.info(f"Content avg length: {avg_content_length}")
        self.__logger.info(f"Content mid length: {mid_content_length}")
        self.__logger.info(f"Content max length: {max(content_length_array)}")
        self.__logger.info(f"Build data size: {len(data_set)}")
        self.__logger.info("Transform raw data finish")

        return data_set

    def image_transform(self, image_url_list, content):

        data_transform = {
            "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "val": transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

        image_list = []
        if content.count("[IMG]") != 0:
            image_url_list = image_url_list[:content.count("[IMG]")]
            for url in image_url_list:
                img = io.imread(url)
                path = hyper_args.get_path()
                io.imsave(path, img)
                image = data_transform['train'](Image.open(path).convert('RGB'))
                image_list.append(image)

        return image_list

    @staticmethod
    def tokenize_one_sample(sample):
        """
        数据处理函数
        Args:
            sample: 一个字典，包含正文,图像，标签，格式为{"content": content, "tags": tags, "image": image}

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
        # 对正文进行tokenizer.tokenize分词

        content = sample["content"]
        tags = sample["tags"]
        image_url_list = sample["image"]

        # 预截断, 加速用
        content_size = hyper_args.N_CTX1 - len(tags) - 3
        content = content[:content_size]

        content_tokens = hyper_args.TOKENIZER.tokenize(content)
        # 对新闻标题进行tokenizer.tokenize分词，注意tokenizer中已经将[Space]作为一个分隔符，不会切割成多个字符
        tags_tokens = hyper_args.TOKENIZER.tokenize(tags)

        # 判断如果正文过长，进行截断
        content_tokens_size = hyper_args.N_CTX - len(tags_tokens) - 3
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

        '''tags'''
        input_ids.extend(hyper_args.TOKENIZER.convert_tokens_to_ids(tags_tokens))
        token_type_ids.extend([hyper_args.TAGS_TYPE_ID] * len(tags_tokens))

        '''image'''
        image_count = 0
        if hyper_args.IMG_ID in input_ids:
            image_feature = []
            image_index = [i for i, x in enumerate(input_ids) if x == hyper_args.IMG_ID]
            image_url_list = image_url_list[:len(image_index)]
            image_count += len(image_index)
            for url in image_url_list:
                try:
                    # img = io.imread(url)
                    response = requests.get(url, proxies=proxies)
                    img = np.asarray(Image.open(BytesIO(response.content)))
                    path = hyper_args.get_path()
                    io.imsave(path, img)
                except Exception as e:
                    return None
                image = data_transform['train'](Image.open(path).convert('RGB'))
                image_feature.append(image)
            for index in image_index:
                token_type_ids[index] = hyper_args.IMG_ID
        else:
            image_feature, image_index = None, None

        '''sep'''
        input_ids.append(hyper_args.TOKENIZER.sep_token_id)
        token_type_ids.append(hyper_args.TAGS_TYPE_ID)

        # 判断input_ids与token_type_ids长度是否一致
        assert len(input_ids) == len(token_type_ids)
        # 判断input_ids长度是否小于等于最大长度
        assert len(input_ids) <= hyper_args.N_CTX
        return input_ids, token_type_ids, image_feature, image_index, image_count

    def tokenize_data(self, data):

        self.__logger.info(f"All cutting size (tokens size): {hyper_args.N_CTX}")

        if hyper_args.CLEAN_TOKENIZE:
            self.__logger.info("Tokenize data begin")

            data_set = []
            image_count = 0
            for idx, sample in tqdm(enumerate(data), desc='Tokenizing data'):
                sample_ids = self.tokenize_one_sample(sample)
                if sample_ids is None:
                    continue
                input_ids, token_type_ids, image_feature, image_index, img_count = sample_ids
                image_count = image_count + img_count
                data_set.append(
                    {"input_ids": input_ids, "token_type_ids": token_type_ids, "image_feature": image_feature,
                     "image_index": image_index})

            self.__logger.info(f"total image : {image_count}")
            self.__logger.info("Tokenize data finish")

            self.__logger.info(f"Cached data in: {hyper_args.DATA_CACHE_PATH}")
            torch.save({"data_set": data_set}, hyper_args.DATA_CACHE_PATH)

        else:
            self.__logger.info("Tokenize data skip")
            self.__logger.info(f"Load tokenize data from cache: {hyper_args.DATA_CACHE_PATH}")
            data_set = torch.load(hyper_args.DATA_CACHE_PATH)["data_set"]
            self.__logger.info(f"Load tokenize data success")

        self.__logger.info(f"Tokenized data size: {len(data_set)}")

        return data_set

    def preparing(self):
        # 从json加载数据
        data = self.data
        self.__logger.info(f"source data size: {len(data)}")
        self.__logger.info("Transform raw data")

        data = self.filter_data(data)

        # show data ana
        if hyper_args.CLEAN_TOKENIZE:
            self.store_tags()

        data = self.build_data(data)

        data = self.tokenize_data(data)

        random.shuffle(data)

        return data


class MyDataSet(torch.utils.data.Dataset):
    """文本关键词生成模型所需要的数据类"""

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
        image_feature_temp = instance['image_feature']
        image_index_temp = instance['image_index']
        # 将input_ids_temp和token_type_ids_temp添加到对应的list中
        input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
        token_type_ids_list.append(torch.tensor(token_type_ids_temp, dtype=torch.long))
        image_feature_list.append(image_feature_temp)
        image_index_list.append(image_index_temp)
    # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
            "token_type_ids": pad_sequence(token_type_ids_list, batch_first=True, padding_value=0),
            "image_feature": image_feature_list, "image_index": image_index_list}


if __name__ == '__main__':
    import logging


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


    _logger = create_logger('test-data')
    # _ = DataPrepare(logger=_logger)
    tags_data = LoadDataFromSource(logger=_logger).load()
    data = DataPrepare(logger=_logger, data=tags_data)
    token = data.tokenize_data(tags_data)
    for x in token:
        if hyper_args.IMG_ID in x['input_ids']:
            print(x)
            exit()

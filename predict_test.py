import torch
import os
import torch.nn.functional as F
import copy
import json
import random
import numpy as np
import torch.backends.cudnn
from pymongo import MongoClient
from tqdm import tqdm
import warnings
from generation.tags.visual_linguistic.model import GPT2LMHeadModel
from generation.tags.visual_linguistic import config as hyper_args
from skimage import io
from torchvision import transforms
from PIL import Image
from io import BytesIO
import requests
import logging

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


class Inference:
    def __init__(self):
        self.set_random_seed()
        self.__device_type = hyper_args.DEVICE_TYPE
        self.__set_device()
        self.__legal_tags = self.__load_legal_tags()
        self.__model = self.__load_model()
        db_uri = 'mongodb://' + 'admin' + ':' + "WS8uHk499Kjf%4s#kzn75G" + '@' + '172.16.8.9' + ':' + '27017'
        client = MongoClient(db_uri, authMechanism='SCRAM-SHA-1', connect=False)
        self.db = client["TheEye"]

    def __set_device(self):
        if self.__device_type == 'cuda':
            # 获取设备信息
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICE"] = hyper_args.DEVICE_IDs

    @staticmethod
    def __load_legal_tags():
        with open(hyper_args.LEGAL_TAGS_PATH, 'r') as f:
            return json.load(f)

    def __load_model(self):
        model = GPT2LMHeadModel.from_pretrained(hyper_args.MODEL_PREDICT_PATH)
        model.to(self.__device_type)
        model.eval()
        return model

    def interact_inference(self):
        # print('开始对文章生成关键词，输入CTRL + C，则退出')
        import time
        while True:
            # content = input("输入的文章正文为:")
            s = time.time()
            new_tags = self.predict_vl()
            e = time.time()
            print(f"生成的关键词为：{new_tags}")
            print(f'耗时：{round(e - s, 4)}s')
            print()

    def predict(self, content):
        tokens = self.__content_to_tokens(content)
        input_ids, token_type_ids = self.__tokens_to_id(tokens)
        input_tensors, token_type_tensors = self.__id_to_tensors(input_ids, token_type_ids)

        tags_set = []
        for i in range(hyper_args.GEN_TIMES):
            raw_tags = self.__inference(input_tensors, token_type_tensors)
            # print(raw_tags)
            for tags in raw_tags:
                tags_set.append(tags)

        new_tags = {}
        for tags in tags_set:
            base = self.__compute_word_weight_base(len(tags))

            for idx, tag in enumerate(tags):
                if tag not in self.__legal_tags:
                    continue
                if tag in new_tags:
                    new_tags[tag] += ((idx + 1) ** -1) / base / len(tags_set)
                else:
                    new_tags[tag] = ((idx + 1) ** -1) / base / len(tags_set)

        new_tags = sorted(new_tags.items(), key=lambda item: item[1], reverse=True)
        # print(new_tags)
        # new_tags = [x[0] for x in new_tags][:hyper_args.MAX_TAGS_NUM]
        return new_tags

    @staticmethod
    def __compute_word_weight_base(nums):
        base = 0
        for i in range(nums):
            base += (i + 1) ** -1

        return base

    @staticmethod
    def __content_to_tokens(content):
        # 预截断加速
        content = content[:hyper_args.N_CTX]
        # tokenize
        content_tokens = hyper_args.TOKENIZER.tokenize(content)

        # cutting
        cutting_size = hyper_args.N_CTX - 3 - hyper_args.MAX_TAGS_LENGTH
        content_tokens = content_tokens[:cutting_size]
        content_tokens = ["[CLS]"] + content_tokens + ["[SEP]"]

        return content_tokens

    @staticmethod
    def __tokens_to_id(tokens):
        # 将tokens索引化，变成模型所需格式
        input_ids = hyper_args.TOKENIZER.convert_tokens_to_ids(tokens)
        # 将input_ids和token_type_ids进行扩充，扩充到需要预测关键词的个数，即batch_size
        input_ids = [copy.deepcopy(input_ids) for _ in range(hyper_args.BATCH_SIZE)]

        # TODO image

        token_type_ids = [[hyper_args.CONTENT_TYPE_ID] * len(tokens) for _ in range(hyper_args.BATCH_SIZE)]

        return input_ids, token_type_ids

    def __id_to_tensors(self, input_ids, token_type_ids):
        # 将input_ids和token_type_ids变成tensor
        input_tensors = torch.tensor(input_ids).long().to(self.__device_type)
        token_type_tensors = torch.tensor(token_type_ids).long().to(self.__device_type)

        return input_tensors, token_type_tensors

    def __inference(self, image_feature, image_index, input_tensors, token_type_tensors):
        """
        对单个样本进行预测
        Args:

        Returns:
        """
        next_token_type_tensor = torch.tensor(
            [[hyper_args.TAGS_TYPE_ID] for _ in range(hyper_args.BATCH_SIZE)]).long().to(self.__device_type)
        # 用于存放每一步解码的结果
        generated = []
        # 用于存放，完成解码序列的序号
        finish_set = set()
        with torch.no_grad():
            # 遍历生成关键词最大长度
            for _ in range(hyper_args.MAX_TAGS_LENGTH):
                outputs = self.__model(image_feature=image_feature, image_index=image_index, input_ids=input_tensors,
                                       token_type_ids=token_type_tensors)

                # 获取预测结果序列的最后一个标记，next_token_logits size：[batch_size, vocab_size]
                next_token_logits = outputs[0][:, -1, :]
                # 对batch_size进行遍历，将词表中出现在序列中的词的概率进行惩罚
                for index in range(hyper_args.BATCH_SIZE):
                    for token_id in set([token_ids[index] for token_ids in generated]):
                        next_token_logits[index][token_id] /= hyper_args.REPETITION_PENALTY
                # 对batch_size进行遍历，将词表中的UNK的值设为无穷小
                for next_token_logit in next_token_logits:
                    next_token_logit[hyper_args.UNK_ID] = -float("Inf")
                # 使用top_k_top_p_filtering函数，按照top_k和top_p的值，对预测结果进行筛选
                filter_logits = self.__top_k_top_p_filtering(next_token_logits, top_k=hyper_args.TOP_K,
                                                             top_p=hyper_args.TOP_P)
                # 对filter_logits的每一行做一次取值，输出结果是每一次取值时filter_logits对应行的下标，即词表位置（词的id）
                # filter_logits中的越大的值，越容易被选中
                next_tokens = torch.multinomial(F.softmax(filter_logits, dim=-1), num_samples=1)
                # 判断如果哪个序列的预测标记为sep_id时，则加入到finish_set
                for index, token_id in enumerate(next_tokens[:, 0]):
                    if token_id == hyper_args.SEP_ID:
                        finish_set.add(index)
                # 判断，如果finish_set包含全部的序列序号，则停止预测；否则继续预测
                finish_flag = True
                for index in range(hyper_args.BATCH_SIZE):
                    if index not in finish_set:
                        finish_flag = False
                        break
                if finish_flag:
                    break
                # 将预测标记添加到generated中
                generated.append([token.item() for token in next_tokens[:, 0]])
                # 将预测结果拼接到input_tensors和token_type_tensors上，继续下一次预测
                input_tensors = torch.cat((input_tensors, next_tokens), dim=-1)
                token_type_tensors = torch.cat((token_type_tensors, next_token_type_tensor), dim=-1)
            # 用于存储预测结果
            candidate_responses = []
            # 对batch_size进行遍历，并将token_id变成对应汉字
            for index in range(hyper_args.BATCH_SIZE):
                responses = []
                for token_index in range(len(generated)):
                    # 判断，当出现sep_id时，停止在该序列中添加token
                    if generated[token_index][index] != hyper_args.SEP_ID:
                        responses.append(generated[token_index][index])
                    else:
                        break
                # 将token_id序列变成汉字序列，并将[Space]替换成空格
                responses_word = "".join(hyper_args.TOKENIZER.convert_ids_to_tokens(responses)) \
                    .replace("[Space]", " ")
                responses_word = responses_word.split(',')
                candidate_responses.append(responses_word)
        return candidate_responses

    @staticmethod
    def __top_k_top_p_filtering(logits, top_k, top_p, filter_value=-float("Inf")):
        """
        top_k或top_p解码策略，仅保留top_k个或累积概率到达top_p的标记，其他标记设为filter_value，后续在选取标记的过程中会取不到值设为无穷小。
        Args:
            logits: 预测结果，即预测成为词典中每个词的分数
            top_k: 只保留概率最高的top_k个标记
            top_p: 只保留概率累积达到top_p的标记
            filter_value: 过滤标记值

        Returns:

        """
        # logits的维度必须为2，即size:[batch_size, vocab_size]
        assert logits.dim() == 2
        # 获取top_k和字典大小中较小的一个，也就是说，如果top_k大于字典大小，则取字典大小个标记
        top_k = min(top_k, logits[0].size(-1))
        # 如果top_k不为0，则将在logits中保留top_k个标记
        if top_k > 0:
            # 由于有batch_size个预测结果，因此对其遍历，选取每个预测结果的top_k标记
            for logit in logits:
                indices_to_remove = logit < torch.topk(logit, top_k)[0][..., -1, None]
                logit[indices_to_remove] = filter_value
        # 如果top_p不为0，则将在logits中保留概率值累积达到top_p的标记
        if top_p > 0.0:
            # 对logits进行递减排序
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            # 对排序后的结果使用softmax归一化，再获取累积概率序列
            # 例如：原始序列[0.1, 0.2, 0.3, 0.4]，则变为：[0.1, 0.3, 0.6, 1.0]
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # 删除累积概率高于top_p的标记
            sorted_indices_to_remove = cumulative_probs > top_p
            # 将索引向右移动，使第一个标记也保持在top_p之上
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            for index, logit in enumerate(logits):
                # 由于有batch_size个预测结果，因此对其遍历，选取每个预测结果的累积概率达到top_p的标记
                indices_to_remove = sorted_indices[index][sorted_indices_to_remove[index]]
                logit[indices_to_remove] = filter_value
        return logits

    @staticmethod
    def set_random_seed():
        """
        设置训练的随机种子
        """
        torch.manual_seed(hyper_args.SEED)
        random.seed(hyper_args.SEED)
        np.random.seed(hyper_args.SEED)
        if hyper_args.DEVICE_TYPE == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def transform_tokens(self, content, image_url_list):

        data_transform = {
            "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "val": transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

        # input_ids = []
        # token_type_ids = []
        # # 预截断, 加速用
        # content_size = hyper_args.N_CTX1 - 3
        # content = content[:content_size]
        #
        # content_tokens = hyper_args.TOKENIZER.tokenize(content)
        #
        # # 判断如果正文过长，进行截断
        # content_tokens_size = hyper_args.N_CTX - 3
        # content_tokens = content_tokens[:content_tokens_size]
        #
        # # 生成模型所需的input_ids和token_type_ids
        # '''cls'''
        # input_ids.append(hyper_args.TOKENIZER.cls_token_id)
        # token_type_ids.append(hyper_args.CONTENT_TYPE_ID)
        #
        # '''content'''
        # input_ids.extend(hyper_args.TOKENIZER.convert_tokens_to_ids(content_tokens))
        # token_type_ids.extend([hyper_args.CONTENT_TYPE_ID] * len(content_tokens))
        #
        # '''sep'''
        # input_ids.append(hyper_args.TOKENIZER.sep_token_id)
        # token_type_ids.append(hyper_args.CONTENT_TYPE_ID)
        #
        # '''image'''
        # if hyper_args.IMG_ID in input_ids:
        #     image_feature = []
        #     image_index = [i for i, x in enumerate(input_ids) if x == hyper_args.IMG_ID]
        #     image_url_list = image_url_list[:len(image_index)]
        #     for url in image_url_list:
        #         try:
        #             # img = io.imread(url)
        #             response = requests.get(url, proxies=proxies)
        #             img = np.asarray(Image.open(BytesIO(response.content)))
        #             path = hyper_args.get_path()
        #             io.imsave(path, img)
        #         except Exception as e:
        #             return None
        #         image = data_transform['val'](Image.open(path).convert('RGB'))
        #         image_feature.append(image)
        #     for index in image_index:
        #         token_type_ids[index] = hyper_args.IMG_ID
        # else:
        #     image_feature, image_index = None, None
        #
        # # 判断input_ids与token_type_ids长度是否一致
        # assert len(input_ids) == len(token_type_ids)
        # # 判断input_ids长度是否小于等于最大长度
        # assert len(input_ids) <= hyper_args.N_CTX
        # input_ids = torch.tensor(input_ids).to(self.__device_type)
        # token_type_ids = torch.tensor(token_type_ids).to(self.__device_type)
        image_feature = []
        image_feature.append(data_transform['val'](
            Image.open('./generation/tags/visual_linguistic/data/image_predict/1.png').convert('RGB')))
        image_feature.append(data_transform['val'](
            Image.open('./generation/tags/visual_linguistic/data/image_predict/2.png').convert('RGB')))
        image_feature.append(data_transform['val'](
            Image.open('./generation/tags/visual_linguistic/data/image_predict/3.png').convert('RGB')))
        image_feature.append(data_transform['val'](
            Image.open('./generation/tags/visual_linguistic/data/image_predict/4.png').convert('RGB')))

        content = '如图[IMG][IMG][IMG][IMG]'
        input_ids = torch.tensor([101, hyper_args.TOKENIZER.convert_tokens_to_ids("如"), hyper_args.TOKENIZER.convert_tokens_to_ids("图"), 13317, 13317, 13317, 13317, 102]).to(self.__device_type)
        token_type_ids = torch.tensor([97, 97, 97, 13317, 13317, 13317, 13317, 97]).to(self.__device_type)
        image_index = [1, 2, 3, 4]

        return content, input_ids, token_type_ids, image_feature, image_index

    def predict_vl(self):
        from generation.tags.visual_linguistic.data_helper import LoadDataFromSource
        data_list = []
        count = 0
        for item in tqdm(self.db['zhihu_answer'].find().limit(100000)):

            if int(item["vote_up_count"]) < 100:
                continue
            content = item['content']

            if len(item['content_pic_urls']) > 0 and len(item['content_pic_urls']) == content.count("[IMG at Here]"):
                content = content.replace("*****", "")
                content = content.replace('[IMG at Here]', "[IMG]")

                image_url_list = item['content_pic_urls']
                try:
                    content, input_ids, token_type_ids, image_feature, image_index = self.transform_tokens(content,
                                                                                                           image_url_list)

                except TypeError:

                    continue
                if image_feature is None:

                    continue
                else:
                    data_list.append([input_ids, token_type_ids, image_feature, image_index, content])
                    break

        tags_set = []
        data = data_list[0]

        for i in range(hyper_args.GEN_TIMES):
            print("content:", data[4])
            from torch.autograd import Variable

            raw_tags = self.__inference(image_feature=[data[2]], image_index=[data[3]],
                                        input_tensors=data[0].unsqueeze(0),
                                        token_type_tensors=data[1].unsqueeze(0))

            for tags in raw_tags:
                tags_set.append(tags)

        new_tags = {}
        for tags in tags_set:
            base = self.__compute_word_weight_base(len(tags))

            for idx, tag in enumerate(tags):
                if tag not in self.__legal_tags:
                    continue
                if tag in new_tags:
                    new_tags[tag] += ((idx + 1) ** -1) / base / len(tags_set)
                else:
                    new_tags[tag] = ((idx + 1) ** -1) / base / len(tags_set)

        new_tags = sorted(new_tags.items(), key=lambda item: item[1], reverse=True)
        # print(new_tags)
        # new_tags = [x[0] for x in new_tags][:hyper_args.MAX_TAGS_NUM]

        return new_tags


def predict_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    new_data = []

    inference_obj = Inference()

    for sample in tqdm(data):
        content = sample['content']
        title = sample['title']
        predict_tags = inference_obj.predict(content)
        new_data.append({'title': title,
                         'content': content,
                         'tags': predict_tags})

    '''save excel file'''
    excel_data = []
    for data in new_data:
        tags = [x[0] for x in data['tags']][:4]
        line = f"{data['title']}\t{tags}\t{data['content']}"
        excel_data.append(line)
    from utils.data_io import save_txt_file
    save_txt_file(excel_data, 'res.excel.txt', end='')


def main():
    # predict_file(hyper_args.PREDICT_FILE)

    _inference_obj = Inference()
    _inference_obj.interact_inference()


if __name__ == '__main__':
    main()

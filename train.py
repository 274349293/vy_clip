import torch
import os
import random
import numpy as np
import torch.utils.data
from torch.utils.data import RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn
import time
from torch import nn
from data_helper import MyDataSet, collate_func, DataPrepare
import config as hyper_args
from model import CLIP
from utils.task_logger import TaskLogger
from utils.data_io import mkdir


class TrainModel:
    def __init__(self, logger, model):
        self.__logger = logger
        self.model = model
        self.set_random_seed()
        self.FP16 = False

    @staticmethod
    def set_random_seed():
        """
        设置训练的随机种子
        """
        torch.manual_seed(hyper_args.SEED)

        random.seed(hyper_args.SEED)
        np.random.seed(hyper_args.SEED)
        np.random.RandomState(hyper_args.SEED)
        if hyper_args.DEVICE_TYPE == torch.device('cuda'):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed(hyper_args.SEED)
            torch.cuda.manual_seed_all(hyper_args.SEED)

    def train(self, train_data, test_data):
        """
        训练模型
        Args:
            train_data: 训练数据类
            test_data: 测试数据类
        Returns:

        """
        tb_write = SummaryWriter()

        # 计算真实的训练batch_size大小
        train_batch_size = int(hyper_args.TRAIN_BATCH_SIZE / hyper_args.GRADIENT_ACCUMULATION_STEPS)
        self.__logger.info(f'True Batch size: {train_batch_size}')
        train_sampler = RandomSampler(train_data)
        train_data_loader = torch.utils.data.DataLoader(train_data, sampler=train_sampler,
                                                        batch_size=train_batch_size, collate_fn=collate_func)
        total_steps = int(len(train_data_loader) * hyper_args.EPOCHS / hyper_args.GRADIENT_ACCUMULATION_STEPS)
        self.__logger.info(f"Total steps: {total_steps}")
        self.model.to(hyper_args.DEVICE_TYPE)
        # 获取模型所有参数
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # 设置优化器
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=hyper_args.LEARNING_RATE, eps=hyper_args.ADAM_EPSILON)

        if self.FP16:
            try:
                from apex import amp
                # from apex.fp16_utils import FP16_Optimizer

            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level="O1")  # 这里是“欧一”，不是“零一”
            # optimizer = FP16_Optimizer(optimizer,
            #                            static_loss_scale=args.static_loss_scale,
            #                            dynamic_loss_scale=args.dynamic_loss_scale,
            #                            dynamic_loss_args={'init_scale': 2 ** 16})

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(hyper_args.WARMUP_PROPORTION * total_steps),
                                                    num_training_steps=total_steps)

        # 清空cuda缓存
        torch.cuda.empty_cache()
        # 将模型调至训练状态
        self.model.train()
        title_id = hyper_args.TAGS_TYPE_ID
        tr_loss, logging_loss, min_loss = 0.0, 0.0, 0.0
        global_step = 0

        # 开始训练模型
        for epoch in trange(0, int(hyper_args.EPOCHS), desc="Epoch", disable=False):
            iter_bar = tqdm(train_data_loader, desc="Iter (loss=X.XXX)", disable=False)
            epoch_avg_loss = 0
            step_count = 0
            for step, batch in enumerate(iter_bar):
                input_ids = batch["input_ids"].to(hyper_args.DEVICE_TYPE if torch.cuda.is_available() else "cpu")
                token_type_ids = batch["token_type_ids"].to(
                    hyper_args.DEVICE_TYPE if torch.cuda.is_available() else "cpu")
                image_feature = batch["input_image"]
                image_index = batch["image_index"]

                # 获取训练结果
                outputs = self.model.forward()
                loss = outputs[0]
                tr_loss += loss.item()
                epoch_avg_loss += loss.item()
                step_count += 1
                # 将损失值放到Iter中，方便观察
                iter_bar.set_description("Iter (loss=%5.3f)" % loss.item())
                # 判断是否进行梯度累积，如果进行，则将损失值除以累积步数
                loss = loss / hyper_args.GRADIENT_ACCUMULATION_STEPS
                if self.FP16:
                    # FP16
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), hyper_args.MAX_GRAD_NORM)
                    # or
                else:
                    # NORMAL
                    # 损失进行回传
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), hyper_args.MAX_GRAD_NORM)
                # 当训练步数整除累积步数时，进行参数优化
                if (step + 1) % hyper_args.GRADIENT_ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

                    # # 如果步数整除logging_steps，则记录学习率和训练集损失值
                    # if hyper_args.LOGGING_STEPS > 0 and global_step % hyper_args.LOGGING_STEPS == 0:
                    #     tb_write.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    #     tb_write.add_scalar("train_loss", (tr_loss - logging_loss) /
                    #                         (hyper_args.LOGGING_STEPS * hyper_args.GRADIENT_ACCUMULATION_STEPS),
                    #                         global_step)
                    #     logging_loss = tr_loss
                    # # 如果步数整除eval_steps，则进行模型测试，记录测试集的损失
                    # if hyper_args.EVAL_STEPS > 0 and global_step % hyper_args.EVAL_STEPS == 0:
                    #     eval_loss = self.evaluate(test_data=test_data)
                    #     tb_write.add_scalar("test_loss", eval_loss, global_step)
                    #     # TODO: eval result
                    #     self.model.train()

                # torch.save(self.model.state_dict(), './')

            # 每个epoch进行完，则保存模型
            self.__logger.info(f"AVG loss: {epoch_avg_loss / step_count}")
            now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            output_dir = os.path.join(hyper_args.MODEL_OUTPUT_PATH, f"checkpoint-{epoch + 1}-{now}")
            model_to_save = self.model.module if hasattr(self.model, "module") else self.model
            model_to_save.save_pretrained(output_dir)
            # 清空cuda缓存
            torch.cuda.empty_cache()

    def evaluate(self, test_data):
        """
        对测试数据集进行模型测试
        Args:
            test_data: 测试数据类
        Returns:

        """
        # 构造测试集的DataLoader
        test_sampler = SequentialSampler(test_data)
        test_data_loader = torch.utils.data.DataLoader(test_data, sampler=test_sampler,
                                                       batch_size=hyper_args.TEST_BATCH_SIZE, collate_fn=collate_func)
        iter_bar = tqdm(test_data_loader, desc="iter", disable=False)
        tags_id = hyper_args.TAGS_TYPE_ID
        total_loss, total = 0.0, 0.0
        # 进行测试
        for step, batch in enumerate(iter_bar):
            # 模型设为eval
            self.model.eval()
            with torch.no_grad():
                input_ids = batch["input_ids"].to(hyper_args.DEVICE_TYPE)
                token_type_ids = batch["token_type_ids"].to(hyper_args.DEVICE_TYPE)
                # 获取预测结果
                outputs = self.model.forward(input_ids=input_ids, token_type_ids=token_type_ids, labels=input_ids,
                                             tags_id=tags_id)
                loss = outputs[0]
                loss = loss.item()
                # 对loss进行累加
                total_loss += loss * len(batch["input_ids"])
                total += len(batch["input_ids"])
        # 计算最终测试集的loss结果
        test_loss = total_loss / total
        return test_loss


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model():
    model_eval = torch.jit.load("model.pt").cuda().eval()
    state_dict = model_eval.state_dict()
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model


def train():
    logger = TaskLogger(task_name='train vl multi tags', log_root=None).root_logger

    # 加载训练数据和测试数据
    data_set = DataPrepare(logger=logger)
    data_set = data_set.tokenized_data
    train_set = data_set[:-1]
    test_set = data_set[-1:]
    train_data = MyDataSet(train_set)
    test_data = MyDataSet(test_set)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 按接口顺序排序显卡序号
    os.environ["CUDA_VISIBLE_DEVICE"] = hyper_args.DEVICE_IDs
    # 实例化GPT2LMHeadModel模型，这里我们没有加载预训练好的模型，而是直接从头开始训练。
    # 判断是否使用预训练好的GPT2模型

    model = build_model()

    # 创建模型的输出目录
    mkdir(hyper_args.MODEL_OUTPUT_PATH)

    # 开始训练
    TrainModel(logger=logger, model=model).train(train_data=train_data, test_data=test_data)


if __name__ == '__main__':
    train()

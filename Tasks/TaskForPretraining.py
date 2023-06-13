import os
import logging
import re
import sys

from transformers import BertTokenizer
from transformers import get_polynomial_decay_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import torch
import time

from model.BasicEncoder import Config
from model.Decoders.MLMDecoder import ForMLMModel
from utils import LoadBertPretrainingDataset, logger_init

sys.path.append('../')


class ModelConfig:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # ========== 数据集相关配置
        self.dataset_dir = os.path.join(self.project_dir, 'data', 'Chinese')
        self.pretrained_model_dir = os.path.join(self.project_dir, "base_chinese")
        self.train_file_path = os.path.join(self.dataset_dir, 'ch.train.txt')
        self.val_file_path = os.path.join(self.dataset_dir, 'ch.valid.txt')
        self.test_file_path = os.path.join(self.dataset_dir, 'ch.test.txt')
        self.data_name = 'Chinese'

        # 如果需要切换数据集，只需要更改上面的配置即可
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_save_dir = os.path.join(self.project_dir, 'saved_models')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.model_save_path = os.path.join(self.model_save_dir, f'model_{self.data_name}')
        self.is_sample_shuffle = True
        self.use_embedding_weight = True
        self.batch_size = 32
        self.max_sen_len = None  # 为None时则采用每个batch中最长的样本对该batch中的样本进行padding
        self.pad_index = 0
        self.random_state = 2023
        self.learning_rate = 4e-5
        self.weight_decay = 0.1
        self.masked_rate = 0.15
        self.log_level = logging.DEBUG
        self.epochs = 50
        self.model_val_per_epoch = 5

        logger_init(log_file_name=self.data_name, log_level=self.log_level,
                    log_dir=self.logs_save_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        bert_config_path = os.path.join(self.pretrained_model_dir, "config.json")
        bert_config = Config.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value
        # 将当前配置打印到日志文件中
        logging.info(" ### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"### {key} = {value}")


def train(config):
    model = ForMLMModel(config)
    last_epoch = -1
    if os.path.exists(config.model_save_path):
        checkpoint = torch.load(config.model_save_path)
        last_epoch = checkpoint['last_epoch']
        loaded_paras = checkpoint['model_state_dict']
        model.load_state_dict(loaded_paras, strict=True)
        # model=torch.load(config.model_save_path)
        logging.info("## 成功载入已有模型，进行追加训练......")
    model = model.to(config.device)
    model.train()
    bert_tokenize = BertTokenizer.from_pretrained(config.pretrained_model_dir).tokenize
    data_loader = LoadBertPretrainingDataset(vocab_path=config.vocab_path,
                                             tokenizer=bert_tokenize,
                                             batch_size=config.batch_size,
                                             max_sen_len=config.max_sen_len,
                                             max_position_embeddings=config.max_position_embeddings,
                                             pad_index=config.pad_index,
                                             is_sample_shuffle=config.is_sample_shuffle,
                                             random_state=config.random_state,
                                             masked_rate=config.masked_rate)
    train_iter, test_iter, val_iter = \
        data_loader.load_train_val_test_data(test_file_path=config.test_file_path,
                                             train_file_path=config.train_file_path,
                                             val_file_path=config.val_file_path)
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
            "initial_lr": config.learning_rate

        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "initial_lr": config.learning_rate
        },
    ]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                          int(len(train_iter) * 0),
                                                          int(config.epochs * len(train_iter)),
                                                          last_epoch=last_epoch)
    max_acc = 0
    state_dict = None
    for epoch in range(config.epochs):
        losses = 0
        start_time = time.time()
        for idx, (b_token_ids, b_mask, b_mlm_label) in enumerate(train_iter):
            b_token_ids = b_token_ids.to(config.device)  # [src_len, batch_size]
            b_mask = b_mask.to(config.device)
            b_mlm_label = b_mlm_label.to(config.device)
            loss, mlm_logits = model(input_ids=b_token_ids,
                                     attention_mask=b_mask,
                                     masked_lm_labels=b_mlm_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses += loss.item()
            mlm_acc, _, _, = accuracy(mlm_logits, b_mlm_label,
                                      data_loader.PAD_IDX)
            if idx % 20 == 0:
                logging.info(f"Epoch: [{epoch + 1}/{config.epochs}], Batch[{idx}/{len(train_iter)}], "
                             f"Train loss :{loss.item():.3f}, Train mlm acc: {mlm_acc:.3f}")
        end_time = time.time()
        train_loss = losses / len(train_iter)
        logging.info(f"Epoch: [{epoch + 1}/{config.epochs}], Train loss: "
                     f"{train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        if (epoch + 1) % config.model_val_per_epoch == 0:
            eval_mlm_acc = evaluate(config, val_iter, model, data_loader.PAD_IDX)
            logging.info(f" ### MLM Accuracy on val: {round(eval_mlm_acc, 4)},")
            # mlm_acc, nsp_acc = evaluate(config, train_iter, model, data_loader.PAD_IDX)
            if eval_mlm_acc > max_acc:
                max_acc = eval_mlm_acc
                torch.save({'last_epoch': scheduler.last_epoch,
                            'model_state_dict': model.state_dict()},
                           config.model_save_path + ".pt")
                torch.save(model, config.model_save_path  + ".pth")


def accuracy(mlm_logits, mlm_labels, PAD_IDX):
    """
    :param mlm_logits:  [src_len,batch_size,src_vocab_size]
    :param mlm_labels:  [src_len,batch_size]
    :param PAD_IDX:
    :return:
    """
    mlm_pred = mlm_logits.transpose(0, 1).argmax(axis=2).reshape(-1)
    # 将 [src_len,batch_size,src_vocab_size] 转成 [batch_size, src_len,src_vocab_size]
    mlm_true = mlm_labels.transpose(0, 1).reshape(-1)
    # 将 [src_len,batch_size] 转成 [batch_size， src_len]
    mlm_acc = mlm_pred.eq(mlm_true)  # 计算预测值与正确值比较的情况，得到预测正确的个数（此时还包括有mask位置）
    mask = torch.logical_not(mlm_true.eq(PAD_IDX))  # 找到真实标签中，mask位置的信息。 mask位置为FALSE，非mask位置为TRUE
    mlm_acc = mlm_acc.logical_and(mask)  # 去掉mlm_acc中mask的部分
    mlm_correct = mlm_acc.sum().item()
    mlm_total = mask.sum().item()
    # print(mlm_correct)
    mlm_acc = float(mlm_correct) / mlm_total
    # print(mlm_acc)
    return [mlm_acc, mlm_correct, mlm_total]


def evaluate(config, data_iter, model, PAD_IDX):
    model.eval()
    mlm_corrects, mlm_totals = 0, 0
    with torch.no_grad():
        for idx, (b_token_ids, b_mask, b_mlm_label) in enumerate(data_iter):
            b_token_ids = b_token_ids.to(config.device)  # [src_len, batch_size]
            b_mask = b_mask.to(config.device)
            b_mlm_label = b_mlm_label.to(config.device)
            mlm_logits = model(input_ids=b_token_ids,
                               attention_mask=b_mask)
            result = accuracy(mlm_logits, b_mlm_label, PAD_IDX)
            _, mlm_cor, mlm_tot = result
            mlm_corrects += mlm_cor
            mlm_totals += mlm_tot
    model.train()
    return float(mlm_corrects) / mlm_totals


def inference(config, sentences=None, masked=False, random_state=None):
    """
    :param config:
    :param sentences:
    :param masked: 推理时的句子是否Mask
    :param language: 语种
    :param random_state:  控制mask字符时的随机状态
    :return:
    """
    bert_tokenize = BertTokenizer.from_pretrained(config.pretrained_model_dir).tokenize
    data_loader = LoadBertPretrainingDataset(vocab_path=config.vocab_path,
                                             tokenizer=bert_tokenize,
                                             pad_index=config.pad_index,
                                             random_state=config.random_state,
                                             masked_rate=0.15)  # 15% Mask掉
    token_ids, pred_idx, mask = data_loader.make_inference_samples(sentences,
                                                                   masked=masked,
                                                                   random_state=random_state)
    model = ForMLMModel(config)
    if os.path.exists(config.model_save_path):
        # checkpoint = torch.load(config.model_save_path)
        # loaded_paras = checkpoint['model_state_dict']
        # model.load_state_dict(loaded_paras)
        model = torch.load(config.model_save_path)
        logging.info("## 成功载入已有模型进行推理......")
    else:
        raise ValueError(f"模型 {config.model_save_path} 不存在！")
    model = model.to(config.device)
    model.eval()
    with torch.no_grad():
        token_ids = token_ids.to(config.device)  # [src_len, batch_size]
        mask = mask.to(config.device)
        mlm_logits = model(input_ids=token_ids,
                           attention_mask=mask)
    pretty_print(token_ids, mlm_logits, pred_idx,
                 data_loader.vocab.itos, sentences)


def pretty_print(token_ids, logits, pred_idx, itos, sentences):
    """
    格式化输出结果
    :param token_ids:   [src_len, batch_size]
    :param logits:  [src_len, batch_size, vocab_size]
    :param pred_idx:   二维列表，每个内层列表记录了原始句子中被mask的位置
    :param itos:
    :param sentences: 原始句子
    :return:
    """
    token_ids = token_ids.transpose(0, 1)  # [batch_size,src_len]
    logits = logits.transpose(0, 1)  # [batch_size, src_len,vocab_size]
    y_pred = logits.argmax(axis=2)  # [batch_size, src_len]
    sep =  ""
    for token_id, sentence, y, y_idx in zip(token_ids, sentences, y_pred, pred_idx):
        sen = [itos[id] for id in token_id]
        sen_mask = sen_mask.replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").lstrip()
        logging.info(f" ### 原始: {sentence}")
        logging.info(f"  ## 掩盖: {sen_mask}")
        for idx in y_idx:
            sen[idx] = itos[y[idx]]
        sen = sen.replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").lstrip()
        logging.info(f"  ## 预测: {sen}")
        logging.info("===============")


if __name__ == '__main__':
    config = ModelConfig()
    train(config)
    # f = open("../4.txt", 'r', encoding='ANSI', )
    # sentences_2 = []
    # n=0
    # for line in f:
    #     sentences_2.append(line)
    #     n+=1
    #     if (n>10):
    #         break
    # inference(config, sentences_2, masked=False, random_state=2023)

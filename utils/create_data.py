import logging
import random
from tqdm import tqdm
from .data_helpers import build_vocab
from .data_helpers import pad_sequence
import torch
from torch.utils.data import DataLoader
import os


def read_ch(filepath=None):
    """
    本函数的作用是格式化原始的数据集
    :param filepath:
    :return:  最终的返回形式为一个list
    """
    with open(filepath, 'r', encoding='ANSI') as f:
        lines = f.readlines()  # 一次读取所有行，每一行为一句
    sentences = []
    for line in tqdm(lines, ncols=80, desc=" ## 正在读取原始数据"):
        print(line)
        sentences.append(line)
    random.shuffle(sentences)  # 将所有段落打乱
    return sentences


def cache(func):
    """
    本修饰器的作用是将数据预处理后的结果进行缓存，下次使用时可直接载入！
    :param func:
    :return:
    """

    def wrapper(*args, **kwargs):
        filepath = kwargs['filepath']
        postfix = kwargs['postfix']
        data_path = filepath.split('.')[0] + '_' + postfix + '.pt'
        if not os.path.exists(data_path):
            logging.info(f"缓存文件 {data_path} 不存在，重新处理并缓存！")
            data = func(*args, **kwargs)
            with open(data_path, 'wb') as f:
                torch.save(data, f)
        else:
            logging.info(f"缓存文件 {data_path} 存在，直接载入缓存文件！")
            with open(data_path, 'rb') as f:
                data = torch.load(f)
        return data

    return wrapper


class LoadBertPretrainingDataset(object):
    r"""

    Arguments:

    """

    def __init__(self,
                 vocab_path='./vocab.txt',
                 tokenizer=None,
                 batch_size=64,
                 max_sen_len=None,
                 max_position_embeddings=512,
                 pad_index=0,
                 is_sample_shuffle=True,
                 random_state=2021,
                 masked_rate=0.15):
        self.tokenizer = tokenizer
        self.vocab = build_vocab(vocab_path)
        self.PAD_IDX = pad_index
        self.SEP_IDX = self.vocab['[SEP]']
        self.CLS_IDX = self.vocab['[CLS]']
        self.MASK_IDS = self.vocab['[MASK]']
        self.a_signal = self.vocab['。']
        self.b_signal = self.vocab['，']
        self.c_signal = self.vocab['“']
        self.batch_size = batch_size
        self.max_sen_len = max_sen_len
        self.max_position_embeddings = max_position_embeddings
        self.pad_index = pad_index
        self.is_sample_shuffle = is_sample_shuffle
        self.masked_rate = masked_rate
        self.random_state = random_state
        random.seed(random_state)

    def get_format_data(self, filepath):
        """
        本函数的作用是将数据集格式化成标准形式
        :param filepath:
        :return:  [sentence 1], [sentence 2],...,[] ]
        """
        return read_ch(filepath)

    def replace_masked_tokens(self, token_ids, candidate_pred_positions, num_mlm_preds):
        """
        本函数的作用是根据给定的token_ids、候选mask位置以及需要mask的数量来返回被mask后的token_ids以及标签信息
        :param token_ids:
        :param candidate_pred_positions:
        :param num_mlm_preds:
        :return:
        """
        pred_positions = []
        mlm_input_tokens_id = [token_id for token_id in token_ids]
        for mlm_pred_position in candidate_pred_positions:
            if len(pred_positions) >= num_mlm_preds:
                break  # 如果已经mask的数量大于等于num_mlm_preds则停止mask
            # 将词替换为['MASK']词元，但这里是直接替换为['MASK']对应的id
            masked_token_id = self.MASK_IDS
            mlm_input_tokens_id[mlm_pred_position] = masked_token_id
            pred_positions.append(mlm_pred_position)  # 保留被mask位置的索引信息
        # 构造mlm任务中需要预测位置对应的正确标签，如果其没出现在pred_positions则表示该位置不是mask位置
        # 则在进行损失计算时需要忽略掉这些位置（即为PAD_IDX）；而如果其出现在mask的位置，则其标签为原始token_ids对应的id
        mlm_label = [self.PAD_IDX if idx not in pred_positions
                     else token_ids[idx] for idx in range(len(token_ids))]
        return mlm_input_tokens_id, mlm_label

    def get_masked_sample(self, token_ids):
        """
        本函数的作用是将传入的 一段token_ids的其中部分进行mask处理
        :param token_ids:         e.g. [101, 1031, 4895, 2243, 1033, 10029, 2000, 2624, 1031,....]
        :return: mlm_input_tokens_id:  [101, 1031, 103, 2243, 1033, 10029, 2000, 103,  1031, ...]
                           mlm_label:  [ 0,   0,   4895,  0,    0,    0,    0,   2624,  0,...]
        """
        candidate_pred_positions = []  # 候选预测位置的索引
        for i, ids in enumerate(token_ids):
            # 在遮蔽语言模型任务中不会预测特殊词元，所以如果该位置是特殊词元
            # 那么该位置就不会成为候选mask位置
            if ids in [self.CLS_IDX, self.SEP_IDX, self.a_signal, self.b_signal, self.c_signal]:
                continue
            candidate_pred_positions.append(i)
            # 保存候选位置的索引， 例如可能是 [ 2,3,4,5, ....]
        random.shuffle(candidate_pred_positions)  # 将所有候选位置打乱，更利于后续随机
        # 被掩盖位置的数量，中默认将15%的Token进行mask
        num_mlm_preds = max(1, round(len(token_ids) * self.masked_rate))
        logging.debug(f" ## Mask数量为: {num_mlm_preds}")
        mlm_input_tokens_id, mlm_label = self.replace_masked_tokens(
            token_ids, candidate_pred_positions, num_mlm_preds)
        return mlm_input_tokens_id, mlm_label

    @cache
    def data_process(self, filepath, postfix='cache'):
        """
        本函数的作用是是根据格式化后的数据制作NSP和MLM两个任务对应的处理完成的数据
        :param filepath:
        :return:
        """
        sentences = self.get_format_data(filepath)
        # 返回的是一个一维列表
        data = []
        max_len = 0
        # 这里的max_len用来记录整个数据集中最长序列的长度，在后续可将其作为padding长度的标准
        desc = f" ## 正在构造MLM样本({filepath.split('.')[1]})"
        for sentence in tqdm(sentences, ncols=80, desc=desc):
            # 遍历每个句子
            logging.debug(f" ## 当前句文本：{sentence}")
            token_s_ids = [self.vocab[token] for token in self.tokenizer(sentence)]
            token_ids = [self.CLS_IDX] + token_s_ids + [self.SEP_IDX]
            if len(token_ids) > self.max_position_embeddings - 1:
                token_ids = token_ids[:self.max_position_embeddings - 1]
                # 模型只取前512个字符
            assert len(token_ids) <= self.max_position_embeddings
            logging.debug(f" ## Mask之前词元结果：{[self.vocab.itos[t] for t in token_ids]}")
            logging.debug(f" ## Mask之前token ids:{token_ids}")
            mlm_input_tokens_id, mlm_label = self.get_masked_sample(token_ids)
            token_ids = torch.tensor(mlm_input_tokens_id, dtype=torch.long)
            mlm_label = torch.tensor(mlm_label, dtype=torch.long)
            max_len = max(max_len, token_ids.size(0))
            logging.debug(f" ## Mask之后token ids:{token_ids.tolist()}")
            logging.debug(f" ## Mask之后词元结果：{[self.vocab.itos[t] for t in token_ids.tolist()]}")
            logging.debug(f" ## Mask之后label ids:{mlm_label.tolist()}")
            logging.debug(f" ## 当前样本构造结束================== \n\n")
            data.append([token_ids, mlm_label])
        all_data = {'data': data, 'max_len': max_len}
        return all_data

    def generate_batch(self, data_batch):
        b_token_ids, b_mlm_label = [], []
        for (token_ids, mlm_label) in data_batch:
            # 开始对一个batch中的每一个样本进行处理
            b_token_ids.append(token_ids)
            b_mlm_label.append(mlm_label)
        b_token_ids = pad_sequence(b_token_ids,  # [batch_size,max_len]
                                   padding_value=self.PAD_IDX,
                                   batch_first=False,
                                   max_len=self.max_sen_len)
        # b_token_ids:  [src_len,batch_size]

        b_mlm_label = pad_sequence(b_mlm_label,  # [batch_size,max_len]
                                   padding_value=self.PAD_IDX,
                                   batch_first=False,
                                   max_len=self.max_sen_len)
        # b_mlm_label:  [src_len,batch_size]

        b_mask = (b_token_ids == self.PAD_IDX).transpose(0, 1)
        # b_mask: [batch_size,max_len]

        return b_token_ids, b_mask, b_mlm_label

    def load_train_val_test_data(self,
                                 train_file_path=None,
                                 val_file_path=None,
                                 test_file_path=None,
                                 only_test=False):
        postfix = f"_ml{self.max_sen_len}_rs{self.random_state}_mr{str(self.masked_rate)[2:]}"
        test_data = self.data_process(filepath=test_file_path,
                                      postfix='test' + postfix)['data']
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=False, collate_fn=self.generate_batch)
        if only_test:
            logging.info(f"## 成功返回测试集，一共包含样本{len(test_iter.dataset)}个")
            return test_iter
        data = self.data_process(filepath=train_file_path, postfix='train' + postfix)
        train_data, max_len = data['data'], data['max_len']
        train_iter = DataLoader(train_data, batch_size=self.batch_size,
                                shuffle=self.is_sample_shuffle,
                                collate_fn=self.generate_batch)
        val_data = self.data_process(filepath=val_file_path, postfix='val' + postfix)['data']
        val_iter = DataLoader(val_data, batch_size=self.batch_size,
                              shuffle=False,
                              collate_fn=self.generate_batch)
        logging.info(f"## 成功返回训练集样本（{len(train_iter.dataset)}）个、开发集样本（{len(val_iter.dataset)}）个"
                     f"测试集样本（{len(test_iter.dataset)}）个.")
        return train_iter, test_iter, val_iter

    def make_inference_samples(self, sentences=None, masked=False, random_state=None):
        """
        制作推理时的数据样本
        :param sentences:
        :param masked:  指传入的句子没有标记mask的位置
        :param random_state:  制作mask字符时的随机状态
        :return:
        e.g.
        sentences = ["I no longer love her, true,but perhaps I love her.",
                     "Love is so short and oblivion so long."]
        input_tokens_ids.transpose(0,1):
                tensor([[  101,  1045,  2053,   103,  2293,  2014,  1010,  2995,  1010,  2021,
                            3383,   103,  2293,  2014,  1012,   102],
                        [  101,  2293,   103,  2061,  2460,  1998, 24034,  2061,  2146,  1012,
                            102,     0,     0,     0,     0,     0]])
        tokens:
                [CLS] i no [MASK] love her , true , but perhaps [MASK] love her . [SEP]
                [CLS] love [MASK] so short and oblivion so long . [SEP] [PAD] [PAD] [PAD] [PAD] [PAD]
        pred_index:
                [[3, 11], [2]]
        mask:
                tensor([[False, False, False, False, False, False, False, False, False, False,
                        False, False, False, False, False, False],
                      [False, False, False, False, False, False, False, False, False, False,
                        False,  True,  True,  True,  True,  True]])
        """
        if not isinstance(sentences, list):
            sentences = [sentences]
        mask_token = self.vocab.itos[self.MASK_IDS]
        input_tokens_ids = []
        pred_index = []
        for sen in sentences:
            if masked:
                sen = sen.replace('[MASK]', '.')
            sen_list = [w for w in sen]
            tmp_token = []
            if not masked:  # 如果传入的样本没有进行mask，则此处进行mask
                candidate_pred_positions = [i for i in range(len(sen_list))]
                random.seed(random_state)
                random.shuffle(candidate_pred_positions)
                num_mlm_preds = max(1, round(len(sen_list) * self.masked_rate))
                for p in candidate_pred_positions[:num_mlm_preds]:
                    sen_list[p] = mask_token
            else:
                num = 0
                for s in sen_list:
                    if s == '.':
                        sen_list[num] = mask_token
                    num += 1
            for item in sen_list:  # 逐个词进行tokenize
                if item == mask_token:
                    tmp_token.append(item)
                else:
                    tmp_token.extend(self.tokenizer(item))
            token_ids = [self.vocab[t] for t in tmp_token]
            token_ids = [self.CLS_IDX] + token_ids + [self.SEP_IDX]
            pred_index.append(self.get_pred_idx(token_ids))
            # 得到被mask的Token的位置
            input_tokens_ids.append(torch.tensor(token_ids, dtype=torch.long))
        input_tokens_ids = pad_sequence(input_tokens_ids,
                                        padding_value=self.PAD_IDX,
                                        batch_first=False,
                                        max_len=None)  # 按一个batch中最长的样本进行padding
        mask = (input_tokens_ids == self.PAD_IDX).transpose(0, 1)
        return input_tokens_ids, pred_index, mask

    def get_pred_idx(self, token_ids):
        """
        根据token_ids返回'[MASK]'所在的位置，即需要预测的位置
        :param token_ids:
        :return:
        """
        pred_idx = []
        for i, t in enumerate(token_ids):
            if t == self.MASK_IDS:
                pred_idx.append(i)
        return pred_idx

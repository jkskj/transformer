import torch.nn as nn
import torch
from torch.nn.init import normal_


class PositionalEmbedding(nn.Module):
    """
    位置编码。
    """

    def __init__(self, embedding_size, max_position_embeddings=256, initializer_range=0.02):
        super(PositionalEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_position_embeddings, embedding_size)
        # 用给定的方式来初始化参数
        self._reset_parameters(initializer_range)

    def forward(self, position_ids):
        """
        :param position_ids: [1,position_ids_len]
        :return: [position_ids_len, 1, hidden_size]
        """
        return self.embedding(position_ids).transpose(0, 1)

    def _reset_parameters(self, initializer_range):
        r"""Initiate parameters."""
        """
        初始化
        """
        for p in self.parameters():
            if p.dim() > 1:
                normal_(p, mean=0.0, std=initializer_range)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, pad_token_id=0, initializer_range=0.02):
        super(TokenEmbedding, self).__init__()
        # padding_idx是用来指定序列中用于padding处理的索引编号，一般来说默认都是0。
        # 在指定padding_idx后，如果输入序列中有0，那么对应位置的向量就会全是0
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_token_id)
        # 用给定的方式来初始化参数
        self._reset_parameters(initializer_range)

    def forward(self, input_ids):
        """
        :param input_ids: shape : [input_ids_len, batch_size]
        :return: shape: [input_ids_len, batch_size, hidden_size]
        """
        return self.embedding(input_ids)

    def _reset_parameters(self, initializer_range):
        r"""Initiate parameters."""
        """
        初始化
        """
        for p in self.parameters():
            if p.dim() > 1:
                normal_(p, mean=0.0, std=initializer_range)


class Embeddings(nn.Module):
    """
    编码层
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = TokenEmbedding(vocab_size=config.vocab_size,
                                              embedding_size=config.embedding_size,
                                              pad_token_id=config.pad_token_id,
                                              initializer_range=config.initializer_range)
        # return shape [src_len,batch_size,hidden_size]

        self.position_embeddings = PositionalEmbedding(max_position_embeddings=config.max_position_embeddings,
                                                       embedding_size=config.embedding_size,
                                                       initializer_range=config.initializer_range)
        # return shape [src_len,1,hidden_size]

        self.LayerNorm = nn.LayerNorm(config.embedding_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids",
                             torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.dense = nn.Linear(config.embedding_size, config.hidden_size)
        # shape: [1, max_position_embeddings]

    def forward(self,
                input_ids=None,
                position_ids=None):
        """
        :param input_ids:  输入序列的原始token id, shape: [src_len, batch_size]
        :param position_ids: 位置序列，本质就是 [0,1,2,3,...,src_len-1], shape: [1,src_len]
        :return: [src_len, batch_size, hidden_size]
        """
        src_len = input_ids.size(0)
        token_embedding = self.word_embeddings(input_ids)
        # shape:[src_len,batch_size,hidden_size]

        if position_ids is None:  # 在实际建模时这个参数其实可以不用传值
            position_ids = self.position_ids[:, :src_len]  # [1,src_len]
        positional_embedding = self.position_embeddings(position_ids)
        # [src_len, 1, hidden_size]

        embeddings = token_embedding + positional_embedding
        # [src_len,batch_size,hidden_size] + [src_len,1,hidden_size]
        embeddings = self.LayerNorm(embeddings)  # [src_len, batch_size, hidden_size]
        embeddings = self.dense(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

import logging
from ..BasicEncoder.Encoder import EncoderModel
from ..BasicEncoder.Encoder import get_activation
import torch.nn as nn
import torch


class ForMLMTransformHead(nn.Module):

    def __init__(self, config, bert_model_embedding_weights=None):
        """
        :param config:
        :param bert_model_embedding_weights:
        the output-weights are the same as the input embeddings, but there is
        an output-only bias for each token. 即TokenEmbedding层中的词表矩阵
        """
        super(ForMLMTransformHead, self).__init__()
        self.dense = nn.Linear(config.embedding_size, config.embedding_size)
        # 用来判断最后分类层中的权重参数是否复用BERT模型Token Embedding中的权重参数，
        # 因为MLM任务最后的预测类别就等于Token Embedding中的各个词，
        # 所以最后分类层中的权重参数可以复用Token Embedding中的权重参数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=1e-12)
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size)
        if bert_model_embedding_weights is not None:
            self.decoder.weight = nn.Parameter(bert_model_embedding_weights)
        # [hidden_size, vocab_size]
        self.decoder.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        """
        :param hidden_states: [src_len, batch_size, hidden_size] Bert最后一层的输出
        :return:
        """
        input_tensor = hidden_states
        hidden_states = self.dense(hidden_states)  # [src_len, batch_size, hidden_size]
        hidden_states = self.transform_act_fn(hidden_states)  # [src_len, batch_size, hidden_size]
        hidden_states = self.LayerNorm(hidden_states + input_tensor)  # [src_len, batch_size, hidden_size]
        hidden_states = self.decoder(hidden_states)
        # hidden_states:  [src_len, batch_size, vocab_size]
        return hidden_states


class ForMLMModel(nn.Module):
    """
    MLM任务
    """

    def __init__(self, config):
        super(ForMLMModel, self).__init__()
        self.encoder = EncoderModel(config)
        weights = None
        if 'use_embedding_weight' in config.__dict__ and config.use_embedding_weight:
            weights = self.encoder.embeddings.word_embeddings.embedding.weight
            logging.info(f"## 使用token embedding中的权重矩阵作为输出层的权重！{weights.shape}")
        self.mlm_prediction = ForMLMTransformHead(config, weights)
        self.config = config

    def forward(self, input_ids,  # [src_len, batch_size]
                attention_mask=None,  # [batch_size, src_len] mask掉padding部分的内容
                position_ids=None,
                masked_lm_labels=None):  # [src_len,batch_size]
        sequence_output, all_encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids)
        # sequence_output = all_encoder_outputs[-1]  # 取Bert最后一层的输出
        # sequence_output: [src_len, batch_size, hidden_size]
        mlm_prediction_logits = self.mlm_prediction(sequence_output)
        # mlm_prediction_logits: [src_len, batch_size, vocab_size]
        if masked_lm_labels is not None:
            loss_fct_mlm = nn.CrossEntropyLoss(ignore_index=0)
            # MLM任务在构造数据集时padding部分和MASK部分都是用的0来填充，所以ignore_index需要指定为0
            mlm_loss = loss_fct_mlm(mlm_prediction_logits.reshape(-1, self.config.vocab_size),
                                    masked_lm_labels.reshape(-1))
            total_loss = mlm_loss
            return total_loss, mlm_prediction_logits
        else:
            return mlm_prediction_logits
        # [src_len, batch_size, vocab_size], [batch_size, 2]

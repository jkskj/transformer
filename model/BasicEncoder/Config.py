import json
import copy
import six
import logging


class Config(object):
    def __init__(self,
                 vocab_size=6195,
                 hidden_size=256,
                 num_hidden_layers=3,
                 num_attention_heads=4,
                 intermediate_size=512,
                 pad_token_id=0,
                 hidden_act="gelu",
                 embedding_size=64,
                 max_position_embeddings=512,
                 initializer_range=0.02):
        self.vocab_size = vocab_size
        self.embedding_size=embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.pad_token_id = pad_token_id
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `Config` from a Python dictionary of parameters."""
        config = Config(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `Config` from a json file of parameters."""
        """从json配置文件读取配置信息"""
        with open(json_file, 'r') as reader:
            text = reader.read()
        logging.info(f"成功导入配置文件 {json_file}")
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

from utils.common import *
from transformers import BertTokenizerFast, BertModel

class StockNLP(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.model_net = BertModel.from_pretrained(model_path)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)

    def forward(self, seq):
        inputs = self.tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model_net(**inputs)
        return outputs.last_hidden_state
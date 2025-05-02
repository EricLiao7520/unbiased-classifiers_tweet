import torch, torch.nn as nn
from transformers import AutoModel
from models import BaseClassifier, DivDisClassifier
class BERT_models_baseline(nn.Module):
    def __init__(self, pretrain_model : str =  "bert-base-uncased",
        mlp_dropout: float = 0.1,
        num_classes: int = 3
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrain_model)
        hidden = self.bert.config.hidden_size
        self.classifier = BaseClassifier(
            input_dim = hidden,
            dropout_rate=mlp_dropout,
            num_classes=num_classes
        )

    def forward(self, input_ids, attn_masks=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attn_masks, token_type_ids=token_type_ids, return_dict=True)
        pooler_outputs = outputs.pooler_output
        logits = self.classifier(pooler_outputs)
        return logits

class BERT_models_DivDis(nn.Module):
    def __init__(
        self, pretrain_model : str =  "bert-base-uncased", 
        num_heads: int = 3,
        mlp_dropout: float = 0.1,
        num_classes: int = 3,
        diversity_weight: float = 1e-3,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrain_model)
        hidden = self.bert.config.hidden_size
        self.divdis = DivDisClassifier(
            input_dim = hidden,
            dropout_rate=mlp_dropout,
            num_heads=num_heads,
            num_classes=num_classes,
            diversity_weight= diversity_weight,
        )
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.diversity_weight = diversity_weight

    def forward(self, input_ids, attn_masks=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attn_masks, token_type_ids=token_type_ids, return_dict=True)
        pooler_outputs = outputs.pooler_output
        logits = self.divdis(pooler_outputs)
        return logits
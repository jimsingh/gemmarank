import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PretrainedConfig, PreTrainedModel, AutoConfig, T5EncoderModel, T5Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, T5TokenizerFast

class RankT5EncConfig(PretrainedConfig):
    model_type = "rankt5enc"
    
    def __init__(self, model_name="t5-large", **kwargs):
        base_config = AutoConfig.from_pretrained(model_name)
        hidden_size = base_config.d_model

        # Remove conflicting keys from kwargs
        dense_lr = kwargs.pop('dense_lr', 1e-4)
        embedding_lr = kwargs.pop('embedding_lr', 5e-6)
        encoder_lr = kwargs.pop('encoder_lr', 1e-4)
        embedding_param_names = kwargs.pop('embedding_param_names', ['shared.weight'])

        super().__init__(**kwargs)
        
        # Set custom attributes after parent init
        self.hidden_size = hidden_size
        self.model_name = model_name
        self.dense_lr = dense_lr
        self.embedding_lr = embedding_lr
        self.encoder_lr = encoder_lr
        self.embedding_param_names = embedding_param_names

    def get_encoder(self):
        return T5EncoderModel.from_pretrained(self.model_name, torch_dtype=torch.bfloat16)


class RankT5Enc(PreTrainedModel):
    config_class = RankT5EncConfig

    def __init__(self, config):
        super().__init__(config)
        self.encoder = config.get_encoder()
        self.dropout = nn.Dropout(0.4)
        self.dense = nn.Linear(config.hidden_size, 1)
        nn.init.normal_(self.dense.weight, std=0.05)
        nn.init.zeros_(self.dense.bias)

    def _get_normalized_scores(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        token_counts = attention_mask.sum(dim=-1, keepdim=True).float()
        pooled = (hidden * attention_mask.unsqueeze(-1).float()).sum(dim=1) / token_counts 
        pooled = self.dropout(pooled)
        scores = self.dense(pooled)
        return scores.clamp(-3.0, 3.0)

    def forward(self, input_ids, attention_mask, labels, result_count):
        # this just reshapes to match encoder expecations
        B, K, L = input_ids.shape
        input_ids = input_ids.view(B * K, L)
        attention_mask = attention_mask.view(B * K, L)

        scores = self._get_normalized_scores(input_ids, attention_mask)
        scores = scores.view(B, K) # non-contiguous memory ... causes a softmax optimization warning

        labels_sum = labels.sum(dim=-1, keepdim=True)
        labels_prob = labels / labels_sum

        log_prob = torch.log_softmax(scores, dim=-1)
        loss = -(labels_prob * log_prob).sum(dim=-1).mean()

        return {'loss': loss}

    def _forward(self, pos_ids=None, pos_mask=None, neg_ids=None, neg_mask=None, **kwargs):
        pos_scores = self._get_normalized_scores(pos_ids, pos_mask)
        neg_scores = self._get_normalized_scores(neg_ids, neg_mask)
        score_diff = pos_scores - neg_scores

        #losses = F.softplus(neg_scores - pos_scores)
        #loss = losses.mean()
        loss = F.margin_ranking_loss(
                pos_scores, neg_scores,
                torch.ones_like(pos_scores),
                1.0)

        return {
            "loss": loss,
            "score_diff": score_diff,
        }

    def predict(self, input_ids, attention_mask):
        return self._get_normalized_scores(input_ids, attention_mask)


def register_rankt5_model():
    AutoConfig.register("rankt5enc", RankT5EncConfig)
    AutoModelForSequenceClassification.register(RankT5EncConfig, RankT5Enc)

register_rankt5_model()


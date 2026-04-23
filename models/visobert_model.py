import torch
import torch.nn as nn
from transformers import AutoModel

class ViSoBERTModel(nn.Module):
    def __init__(self, model_path="vinai/visobert-base", num_labels=3):
        super(ViSoBERTModel, self).__init__()
        self.visobert = AutoModel.from_pretrained(model_path)

        self.dropout_bert = nn.Dropout(0.1)
        self.bert_norm = nn.LayerNorm(2304)
        self.reduce_visobert = nn.Linear(2304, 256)

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_labels)
        )

    def forward(self, input_ids, attention_mask, char_input=None):
        outputs = self.visobert(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Concat 3 lớp cuối: [-1], [-2], [-3] -> [B, S, 2304]
        all_layers = outputs.hidden_states
        bert_out = torch.cat((all_layers[-1], all_layers[-2], all_layers[-3]), dim=-1)
        bert_out = self.dropout_bert(bert_out)
        bert_out = self.bert_norm(bert_out)
        
        # Hạ chiều về 256
        bert_out = torch.relu(self.reduce_visobert(bert_out)) # [B, S, 256]
        
        # ================= Pooling =================
        # Mean pooling (masked)
        mask = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(bert_out * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        # Max pooling (masked)
        mask_bool = attention_mask.unsqueeze(-1).bool()
        bert_masked = bert_out.masked_fill(~mask_bool, -1e9)
        max_pooled = torch.max(bert_masked, dim=1).values
        
        # concat
        pooled = torch.cat([mean_pooled, max_pooled], dim=-1) # [B, 512]

        # ================= Classifier =================
        logits = self.classifier(pooled)

        return logits
import torch
import torch.nn as nn
from transformers import AutoModel

class PhobertCharCNNModel(nn.Module):
    def __init__(self, phobert_path, char_vocab_size):
        super().__init__()
        
        # ===== PhoBERT =====
        self.phobert = AutoModel.from_pretrained(phobert_path)
        self.dropout_bert = nn.Dropout(0.1) 
        self.bert_norm = nn.LayerNorm(2304)
        self.reduce_phobert = nn.Linear(2304, 256)

        # ===== CharCNN =====
        self.char_embedding = nn.Embedding(char_vocab_size, 50, padding_idx=0)
        # Multi-scale CNN (kernel 2,3,4,5)
        self.convs = nn.ModuleList([
            nn.Conv1d(50, 64, kernel_size=k) 
            for k in [2, 3, 4, 5]
        ])
        # Nén output CNN (64*4 = 256) về 128
        self.char_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # ===== Feature Fusion Layer =====
        # Đầu ra sau khi nối là vector: [B, S, 256 + 128] = [B, S, 384d]
        self.fusion_dim = 256 + 128

        # Thêm 1 lớp Linear để nén 384 -> 256, giữ kích thước đồng bộ
        self.match_dim_fc = nn.Linear(self.fusion_dim, 256)
        self.post_match_norm = nn.LayerNorm(256)

        # ===== 2. Pre-Classifier Fully Connected Layer =====
        # Bây giờ Pooling sẽ là 256 * 2 = 512 (Giống hệt các model khác)
        self.pooled_dim = 512 
        self.pre_classifier_fc = nn.Linear(self.pooled_dim, 128)
        self.fc_dropout = nn.Dropout(0.3)

        # ===== Final Classification Layer (Softmax) =====
        self.classifier = nn.Linear(128, 3) # 3 nhãn

    def forward(self, input_ids, attention_mask, char_input):
        mask_expanded = attention_mask.unsqueeze(-1).float()
        
        # --- PhoBERT Branch ---
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        all_layers = outputs.hidden_states
        bert_out = torch.cat((all_layers[-1], all_layers[-2], all_layers[-3]), dim=-1)
        bert_out = self.dropout_bert(bert_out)
        bert_out = self.bert_norm(bert_out)
        bert_out = torch.relu(self.reduce_phobert(bert_out)) # [B, S, 256d]
        bert_out = bert_out * mask_expanded 

        # --- CharCNN Branch ---
        B, S, W = char_input.shape
        char_in = char_input.view(-1, W) 
        char_emb = self.char_embedding(char_in).transpose(1, 2)

        char_conv_outs = []
        for conv in self.convs:
            c = torch.relu(conv(char_emb))
            c = torch.max(c, dim=2)[0]
            char_conv_outs.append(c)

        char_feat = torch.cat(char_conv_outs, dim=1) 
        char_feat = self.char_fc(char_feat).view(B, S, 128) # [B, S, 128d]
        char_feat = char_feat * mask_expanded

        # --- Naive Feature Fusion ---
        concat_feat = torch.cat([bert_out, char_feat], dim=-1) # Output: [B, S, 384d]

        # Nén qua Linear thay vì BiLSTM để đồng bộ chiều
        matched_feat = torch.relu(self.match_dim_fc(concat_feat)) # [B, S, 256d]
        matched_feat = self.post_match_norm(matched_feat)
        matched_feat = matched_feat * mask_expanded

        # ================= Pooling =================
        # Mean pooling
        sum_embeddings = torch.sum(matched_feat, dim=1) 
        sum_mask = torch.clamp(attention_mask.sum(dim=1, keepdim=True).float(), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        # Max pooling
        mask_bool = attention_mask.unsqueeze(-1).bool()
        matched_feat_masked = matched_feat.masked_fill(~mask_bool, -1e9)
        max_pooled = torch.max(matched_feat_masked, dim=1)[0]

        # Nối lại thành vector 512 chiều [B, 512]
        pooled = torch.cat([mean_pooled, max_pooled], dim=-1)

        # --- Pre-Classifier Fully Connected Layer ---
        fc_out = torch.relu(self.pre_classifier_fc(pooled))
        fc_out = self.fc_dropout(fc_out)

        # --- Final Classification Layer ---
        return self.classifier(fc_out)
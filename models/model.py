import torch
import torch.nn as nn
from transformers import AutoModel

class HybridHateSpeechModel(nn.Module):
    def __init__(self, phobert_path, char_vocab_size, hidden_dim=128):
        super().__init__()
        
        # ===== PhoBERT =====
        self.phobert = AutoModel.from_pretrained(phobert_path)
        self.dropout_bert = nn.Dropout(0.1) 
        self.bert_norm = nn.LayerNorm(2304)
        # Hạ chiều PhoBERT từ 2304 xuống 256 
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
        # ===== 1. Feature Fusion Layer (Naive Concatenation) =====
        # Đầu vào của BiLSTM là vector nối [B, S, 256 + 128] = [B, S, 384d]
        self.input_dim = 256 + 128

        # ===== 2. BiLSTM Layer =====
        # Sơ đồ hiển thị lớp này hoạt động trên vector nối.
        self.bilstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True
        )
        self.post_lstm_norm = nn.LayerNorm(hidden_dim * 2)
        self.post_lstm_dropout = nn.Dropout(0.2)

        # ===== 3. Pre-Classifier Fully Connected Layer (FC Layer) =====
        # Sơ đồ hiển thị một Lớp FC rõ ràng sau BiLSTM.
        # Chúng tôi sẽ áp dụng pooling trước để tạo ra một vector đại diện toàn câu [B, 512d]
        self.pooled_dim = 256 * 2 # hidden_dim * 4 (mean + max pooling) = 512d
        self.pre_classifier_fc = nn.Linear(self.pooled_dim, 128)
        self.fc_dropout = nn.Dropout(0.3)

        # ===== 4. Final Classification Layer (Softmax) =====
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

        # --- 1. Naive Feature Fusion ---
        # Nối đặc trưng thô đúng như sơ đồ hiển thị
        concat_feat = torch.cat([bert_out, char_feat], dim=-1) # [B, S, 384d]

        # --- 2. BiLSTM Layer ---
        # Học ngữ cảnh trên vector nối thô
        lstm_out, _ = self.bilstm(concat_feat) # Output: [B, S, 256d]
        lstm_out = self.post_lstm_norm(lstm_out)
        lstm_out = lstm_out * mask_expanded
        lstm_out = self.post_lstm_dropout(lstm_out)

        # ================= Pooling =================
        # Mean pooling
        sum_embeddings = torch.sum(lstm_out, dim=1) 
        sum_mask = torch.clamp(attention_mask.sum(dim=1, keepdim=True).float(), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        # Max pooling
        mask_bool = attention_mask.unsqueeze(-1).bool()
        lstm_out_masked = lstm_out.masked_fill(~mask_bool, -1e9)
        max_pooled = torch.max(lstm_out_masked, dim=1)[0]

        # Nối lại thành vector 512 chiều [B, 512d]
        pooled = torch.cat([mean_pooled, max_pooled], dim=-1)

        # --- 3. Pre-Classifier Fully Connected Layer ---
        # Sơ đồ hiển thị Lớp FC này hoạt động trên vectorpooled (512d -> 128d)
        fc_out = torch.relu(self.pre_classifier_fc(pooled))
        fc_out = self.fc_dropout(fc_out)

        # --- 4. Final Classification Layer ---
        # Phân loại cuối cùng (128d -> 3 labels)
        return self.classifier(fc_out)
import torch
from torch.utils.data import Dataset
from pyvi import ViTokenizer

class ViHSDDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, char_to_idx, max_char_per_word=15):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.char_to_idx = char_to_idx
        self.max_char_per_word = max_char_per_word

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        # 1. Tách từ cho BERT
        text_segmented = ViTokenizer.tokenize(text)
        encoding = self.tokenizer(
            text_segmented,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )

        # 2. Xử lý Char-level cho TỪNG TOKEN
        # Lấy các token đã tách (tokens)
        tokens = text_segmented.split()[:self.max_len]

        char_input = []
        for token in tokens:
            # Loại bỏ dấu gạch dưới của ViTokenizer (ví dụ: học_sinh -> học sinh) 
            # để CharCNN học đặc trưng ký tự thuần túy
            clean_token = token.replace("_", "")
            ids = [self.char_to_idx.get(c, 1) for c in clean_token[:self.max_char_per_word]]
            ids += [0] * (self.max_char_per_word - len(ids))
            char_input.append(ids)
            
        # Padding cho đủ số lượng từ (max_len)
        while len(char_input) < self.max_len:
            char_input.append([0] * self.max_char_per_word)
        
        char_input = torch.tensor(char_input[:self.max_len], dtype=torch.long)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'char_input': char_input, # Kết quả là [Max_Len, Max_Char_per_word]
            'label': torch.tensor(label, dtype=torch.long)
        }
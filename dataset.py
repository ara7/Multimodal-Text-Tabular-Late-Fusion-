# dataset.py
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import config

class MultimodalDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = df[config.TEXT_FEATURE]
        self.tabular = df[config.TABULAR_FEATURES].values.astype(float) 
        self.labels = df[config.LABEL_COLUMN].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tabular_data = self.tabular[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'tabular_data': torch.tensor(tabular_data, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long)
        }

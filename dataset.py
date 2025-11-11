# Author: Ara, Lena
# Description: Custom PyTorch Dataset class for loading and preprocessing
# the multimodal (tabular + text) patient data.

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import config

class MultimodalDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = df[config.TEXT_FEATURE].values
        self.tabular_1 = df[config.TABULAR_FEATURES_1].values.astype(float)
        self.tabular_2 = df[config.TABULAR_FEATURES_2].values.astype(float)
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
        tabular_data_1 = self.tabular_1[idx]
        tabular_data_2 = self.tabular_2[idx]

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'tabular_data_1': torch.tensor(tabular_data_1, dtype=torch.float),
            'tabular_data_2': torch.tensor(tabular_data_2, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long)
        }

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class MedicalTextDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=512):
        self.data_frame = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        text = str(self.data_frame.iloc[idx, 0])
        label = int(self.data_frame.iloc[idx, 1])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        sample = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'grade_label': torch.tensor(label, dtype=torch.long)
        }
        
        return sample

def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)
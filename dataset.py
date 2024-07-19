import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class ToxicityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, stride):
        # Intialize model
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.stride = stride

    # Return length of dataset
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Get the text and label 
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize the text 
        encodings = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_overflowing_tokens=True,
            stride=self.stride,
            return_tensors='pt'
        )
        
        # Extract input IDs and attention mask from the tokenized output
        input_ids = encodings.input_ids.squeeze(0)
        attention_mask = encodings.attention_mask.squeeze(0)
        
        #  Extract seg ids if exists
        if 'token_type_ids' in encodings:
            seg_ids = encodings.token_type_ids.squeeze(0)
        else:
            seg_ids = torch.zeros_like(input_ids)  
        
        # Return the features needed for training
        return {
            'text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'seg_ids': seg_ids,
            'label': torch.tensor(label, dtype=torch.long)
        }

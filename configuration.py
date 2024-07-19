import torch
from transformers import BertTokenizer, BertModel
import pickle
from pathlib import Path

class Config:
    def __init__(self, config_dir, model_dir):
        # Bert Config
        self.vocab_size = 30522  
        self.max_len = 512 
        self.n_layers = 12
        self.d_model = 768
        self.d_ff = 3072
        self.n_heads = 12
        self.drop_p = 0.1
        self.batch_size = 16
        self.epochs = 3
        self.lr = 2e-5
        self.stride = 256
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_path = 'the-5th-kaggle-competition-challenge-with-kakr/train.csv'
        self.num_classes = 11
        self.global_epoch = 0
        self.accumulation_steps = 2
        
        # Dir Config
        self.dir_path = Path(config_dir)
        self.dir_path.mkdir(exist_ok=True)

        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        # Tokenizer config
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.pad_idx = self.tokenizer.pad_token_id

    def __reduce__(self):
        return (self.__class__, (self.config_dir, self.model_dir))
    
    # Save config
    def save(self):
        save_path = self.config_dir / 'config.pkl'
        with open(save_path, 'wb') as file:
            pickle.dump(self, file)

    # Load config
    @staticmethod
    def load(dir_path):
        load_path = Path(dir_path) / 'config.pkl'
        with open(load_path, 'rb') as file:
            config  = pickle.load(file)
        return config
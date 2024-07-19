import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from dataset import ToxicityDataset
import data_process as process
from configuration import Config
from model import BERT, BERTForToxicity, LoRA
import torch.nn as nn
import gc
from tqdm import tqdm
import os
from pathlib import Path
from torch.optim.lr_scheduler import StepLR


# Set up the seed
def seed_set(SEED = 42):    
    torch.manual_seed(SEED)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    print(f'Seed set at {SEED}')
    
# Collect the computing unit and empty memory
def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
# Check if a given direcoty is empty
def is_directory_empty(dir_path):
    path = Path(dir_path)
    return not any(path.iterdir())

# Load train and validation dataloaders
def load_dataloaders(csv_path, config):
    
    # Get the dataframe
    df = process.get_dataframe(csv_path)

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Split the dataset into test and validaton
    split_idx = int(len(df) * 0.8)
    train_df = df[:split_idx]
    val_df = df[split_idx:]

    # Create datasets
    train_dataset = ToxicityDataset(
        texts=train_df["comment_text"].tolist(),
        labels=train_df["toxicity"].tolist(),
        tokenizer=tokenizer,
        max_len=config.max_len,
        stride=config.stride,
    )

    val_dataset = ToxicityDataset(
        texts=val_df["comment_text"].tolist(),
        labels=val_df["toxicity"].tolist(),
        tokenizer=tokenizer,
        max_len=config.max_len,
        stride=config.stride,
    )

    # Create dataloaders and return them
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )

    return train_dataloader, val_dataloader



def get_model(config):
    print('Loading BERT')
    
    # Create custom BERT
    bert = BERT(config).to(config.device)
    # Load pretrained weights
    bert.load_pretrained_weights("bert-base-uncased")

    # Apply LoRA adaptation with rank=10
    for name, module in bert.named_modules():
        if isinstance(module, nn.Linear) and any(
            attn in name for attn in ["fc_q", "fc_k", "fc_v", "fc_o"]
        ):
            parent_module = bert
            for attr in name.split(".")[:-1]:
                parent_module = getattr(parent_module, attr)
            lora_module = LoRA(module, rank=10)
            setattr(parent_module, name.split(".")[-1], lora_module)

    # Selectively freeze layers
    for name, param in bert.named_parameters():
        if (
            "encoder.token_embedding" in name
            or "encoder.pos_embedding" in name
            or "encoder.seg_embedding" in name
            or "encoder.layers.0" in name
            or "encoder.layers.1" in name
            or "encoder.layers.2" in name
            or "encoder.layers.3" in name
            or "encoder.layers.4" in name
        ):
            param.requires_grad = False

    model = BERTForToxicity(bert, config.d_model, config.num_classes).to(config.device)

    # Print out parameters

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    return model


def training_and_validation(model, loss_fn, optimizer, scheduler, train_dataloader, val_dataloader, config, load=False):
    # Intilaize best-loss and best_model path
    best_loss = float('inf')
    best_model_path = config.model_dir / f'best_model.pth'
    
    #vGet accumilation steps from configuration
    accumulation_steps = config.accumulation_steps  
    
    # Load the best model if it exsits 
    if load == True and best_model_path.exists():
        model.state_dict(torch.load(best_model_path))
    
    # Train the model
    for epoch in range(config.epochs):
        model.train()
        # Set up variable for loss  and accuracy calculation
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Create train batch iteractor
        train_batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {config.global_epoch + 1} [Train]")

        for i, batch in enumerate(train_batch_iterator):
            # Get all the instances needed for training
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            seg_ids = batch['seg_ids'].to(config.device)
            labels = batch['label'].to(config.device)

            # Get predicton from the model
            y_pred = model(input_ids, attention_mask, seg_ids)
            # Calculate the loss
            loss = loss_fn(y_pred, labels)
            loss /= accumulation_steps
            # Backward loss
            loss.backward()

            # Step the optimizer for certian  accumulation steps
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps 

            # Count number of right predictions
            _, predicted = torch.max(y_pred, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
            
            train_batch_iterator.set_postfix(batch_loss=loss.item())

        # Calculate average training loss and accuracy
        train_loss /= len(train_dataloader)
        train_accuracy = 100 * correct_train / total_train
        scheduler.step()

        model.eval()
        
        # Set up variable for loss  and accuracy calculation
        test_loss = 0.0
        correct_val = 0
        total_val = 0
        
        # Create validation iteractor
        val_batch_iterator = tqdm(val_dataloader, desc=f"Processing Epoch {config.global_epoch + 1} [Validation]")
        with torch.inference_mode():
            for batch in val_batch_iterator:
                # Get all the instances needed 
                input_ids = batch['input_ids'].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                seg_ids = batch['seg_ids'].to(config.device)
                labels = batch['label'].to(config.device)

                # Get predicton from the model
                y_pred = model(input_ids, attention_mask, seg_ids)
                # Calciulate the loss
                loss = loss_fn(y_pred, y)
                test_loss += loss.item()
                
                # Count number of right predictions
                _, predicted = torch.max(y_pred, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
                
                val_batch_iterator.set_postfix(batch_loss=loss.item())

        test_loss /= len(val_dataloader)
        val_accuracy = 100 * correct_val / total_val
        
        # Printout stats
        print(f"Epoch {config.global_epoch + 1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Loss: {test_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Save the model at the end of each epoch
        epoch_model_path = os.path.join(config.model_dir, f'model_epoch_{config.global_epoch + 1:02d}.pth')
        torch.save(model.state_dict(), epoch_model_path)
        config.global_epoch += 1 

        # Update best model if validation loss improves
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), best_model_path)
        
        # Step the scheulder
        scheduler.step()
        
    # Save the updated configuration
    config.save()


if __name__ == "__main__":
    
    csv_path = 'the-5th-kaggle-competition-challenge-with-kakr/train.csv'

    # Prepare  for training
    flush()
    seed_set()
    config = Config('config', 'weights')
    
    # Load if previous configuration exists
    if not is_directory_empty('config'):
        config.load('config')
    
    # Get instances needed 
    train_dataloader, val_dataloader = load_dataloaders(csv_path, config)
    model = get_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = StepLR(optimizer, step_size = 1, gamma=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()
    # Train and validate the model
    training_and_validation(model, loss_fn, optimizer, scheduler, train_dataloader, val_dataloader, config, load=True)

    



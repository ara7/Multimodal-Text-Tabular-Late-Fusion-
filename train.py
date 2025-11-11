# Author: Ara, Lena
# Description: Main script for training the multimodal fusion classifier.
# This script orchestrates data loading, model initialization,
# training, and evaluation.

import pandas as pd
import numpy as np
import random
import torch
import time
import os
import argparse

import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

import config
from dataset import MultimodalDataset
from model import BertClassifier

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPUs available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    print(f"Set seed to {seed_value}")

def save_model(model, epoch):
    # Ensure the save directory exists
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)

    save_filename = f'model_epoch_{epoch}.pth'
    save_path = os.path.join(config.MODEL_SAVE_PATH, save_filename)

    # Best practice: save the model's state_dict, not the entire model
    torch.save(model.state_dict(), save_path)
    print(f"Model checkpoint saved to {save_path}")

def initialize_model(train_dataloader):
    #Creates the model, optimizer, and scheduler.
    print("Initializing model...")
    bert_classifier = BertClassifier(freeze_bert=False)
    bert_classifier.to(device)

    optimizer = AdamW(
        bert_classifier.parameters(),
        lr=config.LEARNING_RATE,
        eps=1e-8
    )

    total_steps = len(train_dataloader) * config.EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    return bert_classifier, optimizer, scheduler

def evaluate(model, val_dataloader, loss_fn):
    """Runs one full pass over the validation set."""

    model.eval()  # Evaluation mode
    val_loss = []
    val_accuracy = []

    for batch in val_dataloader:
        # Move batch to device
        b_input_ids = batch['input_ids'].to(device)
        b_attn_mask = batch['attention_mask'].to(device)
        b_tabular = batch['tabular_data'].to(device)
        b_labels = batch['label'].to(device)

        # Compute logits with no gradient
        with torch.no_grad():
            logits, _, _ = model(b_input_ids, b_attn_mask, b_tabular)

        # Calculate loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1).flatten()
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute average loss and accuracy
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy

def train_loop(model, optimizer, scheduler, train_dataloader, val_dataloader, loss_fn):

    print("========  Starting Training  ========")
    for epoch_i in range(config.EPOCHS):
        print(f"\n{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-" * 60)

        t0_epoch, t0_batch = time.time(), time.time()
        total_loss, batch_loss, batch_counts = 0, 0, 0

        model.train()  # Training mode

        for step, batch in enumerate(train_dataloader):
            batch_counts += 1

            # Move batch to device
            b_input_ids = batch['input_ids'].to(device)
            b_attn_mask = batch['attention_mask'].to(device)
            b_tabular = batch['tabular_data'].to(device)
            b_labels = batch['label'].to(device)

            # Zero out gradients
            model.zero_grad()

            # Forward pass
            logits, _, _ = model(b_input_ids, b_attn_mask, b_tabular)

            # Compute loss
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Backward pass
            loss.backward()

            # Clip gradient to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters
            optimizer.step()
            scheduler.step()

            # --- Validation step (every 40 batches) ---
            if (step % 40 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch

                # Run validation
                val_loss, val_accuracy = evaluate(model, val_dataloader, loss_fn)

                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f}% | {time_elapsed:^9.2f}s")

                # Reset batch timer and loss
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # ---Summary ---
        avg_train_loss = total_loss / len(train_dataloader)
        epoch_time = time.time() - t0_epoch
        val_loss, val_accuracy = evaluate(model, val_dataloader, loss_fn)

        print("-" * 60)
        print(f"Epoch {epoch_i + 1} summary:")
        print(f"  Average Train Loss: {avg_train_loss:.6f}")
        print(f"  Validation Loss:    {val_loss:.6f}")
        print(f"  Validation Accuracy: {val_accuracy:.2f}%")
        print(f"  Epoch Time:         {epoch_time:.2f}s")
        print("-" * 60)

        # Save model checkpoint
        save_model(model, epoch_i + 1)

    print("\n======== Training Complete ========")

def main(args):
    #Main execution function.

    # Override config epochs if provided via command line
    if args.epochs:
        config.EPOCHS = args.epochs
        print(f"Running for {config.EPOCHS} epochs.")

    # Set seed for reproducibility
    set_seed(config.SEED)

    # --- 1. Load Data ---
    print("Loading data...")
    df_train_full = pd.read_csv(config.TRAIN_DATA).dropna(subset=[config.TEXT_FEATURE])

    # Create a train/validation split (e.g., 90% train, 10% val)
    # This is crucial for monitoring overfitting
    df_train, df_val = train_test_split(
        df_train_full,
        test_size=0.1,
        random_state=config.SEED,
        stratify=df_train_full[config.LABEL_COLUMN]
    )
    print(f"Full training data: {len(df_train_full)} samples")
    print(f"Split into {len(df_train)} train and {len(df_val)} validation samples")

    # --- 2. Create Datasets and DataLoaders ---
    print("Initializing tokenizer and datasets...")
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME, do_lower_case=True)

    train_dataset = MultimodalDataset(
        df=df_train,
        tokenizer=tokenizer,
        max_len=config.MAX_LEN
    )

    val_dataset = MultimodalDataset(
        df=df_val,
        tokenizer=tokenizer,
        max_len=config.MAX_LEN
    )

    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=config.BATCH_SIZE
    )

    val_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=config.BATCH_SIZE
    )

    # --- 3. Initialize Model and Loss ---
    model, optimizer, scheduler = initialize_model(train_dataloader)

    # Loss function (CrossEntropyLoss is good for classification)
    loss_fn = nn.CrossEntropyLoss()

    # --- 4. Start Training ---
    train_loop(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the multimodal classifier")
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs to override config.py"
    )
    args = parser.parse_args()

    main(args)

# manual_train.py
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AdamW,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os

def manual_fine_tune():
    print("Starting manual fine-tuning...")
    
    # Load dataset
    dataset = load_dataset("ag_news")
    print("Dataset loaded")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print("Tokenizer loaded")
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding=True, 
            max_length=128
        )
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")
    print("Dataset tokenized")
    
    # Data loaders
    train_loader = DataLoader(tokenized_dataset["train"], batch_size=16, shuffle=True)
    eval_loader = DataLoader(tokenized_dataset["test"], batch_size=16)
    print("Data loaders created")
    
    # Model
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    print("Model loaded")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_epochs = 2
    num_training_steps = num_epochs * len(train_loader)
    
    # Training loop
    model.train()
    print("Starting training loop...")
    
    for epoch in range(num_epochs):
        total_loss = 0
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 500 == 0:
                print(f"Epoch {epoch + 1}, Batch {i}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
    
    # Evaluation
    model.eval()
    predictions = []
    true_labels = []
    
    print("Starting evaluation...")
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch["labels"].cpu().numpy())
            
            if i % 100 == 0:
                print(f"Evaluated {i * 16} samples")
    
    # Calculate metrics
    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="weighted")
    
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save model
    os.makedirs("./bert_ag_news_final", exist_ok=True)
    model.save_pretrained("./bert_ag_news_final")
    tokenizer.save_pretrained("./bert_ag_news_final")
    print("Model saved successfully!")
    
    return model, tokenizer

if __name__ == "__main__":
    manual_fine_tune()
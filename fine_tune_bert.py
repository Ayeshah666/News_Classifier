# fine_tune_bert.py
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
def load_ag_news():
    dataset = load_dataset("ag_news")
    return dataset

# Preprocess and tokenize
def preprocess_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding=True, 
            max_length=128
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

# Compute metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    
    return {"accuracy": acc, "f1_score": f1}

# Fine-tune BERT
def fine_tune_bert():
    # Load dataset
    dataset = load_ag_news()
    
    # Initialize tokenizer and model
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Get label names
    label_names = ["World", "Sports", "Business", "Sci/Tech"]
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=4,
        id2label={i: label for i, label in enumerate(label_names)},
        label2id={label: i for i, label in enumerate(label_names)}
    )
    
    # Tokenize dataset
    tokenized_dataset = preprocess_dataset(dataset, tokenizer)
    
    # Try different parameter combinations for old versions
    try:
        # Try with minimal parameters first
        training_args = TrainingArguments(
            output_dir="./bert_ag_news",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=2,
            weight_decay=0.01,
            logging_steps=500,
            save_steps=500,
            eval_steps=500,
        )
    except TypeError:
        try:
            # Try even simpler
            training_args = TrainingArguments(
                output_dir="./bert_ag_news",
                learning_rate=2e-5,
                per_device_train_batch_size=16,
                num_train_epochs=2,
                logging_steps=500,
            )
        except TypeError:
            # Most basic version
            training_args = TrainingArguments(
                output_dir="./bert_ag_news",
                per_device_train_batch_size=16,
                num_train_epochs=2,
            )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Evaluate
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")
    
    # Save model
    trainer.save_model("./bert_ag_news_final")
    tokenizer.save_pretrained("./bert_ag_news_final")
    
    return trainer, tokenizer, label_names

# Run fine-tuning
if __name__ == "__main__":
    fine_tune_bert()
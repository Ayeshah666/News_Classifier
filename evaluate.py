# evaluate.py
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import os

class NewsClassifier:
    def __init__(self, model_path="./bert_ag_news_final"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Convert relative path to absolute path
        model_path = os.path.abspath(model_path)
        
        # Check if model exists locally
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}. Please train the model first.")
        
        # Load tokenizer and model from local directory
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Label mapping (AG News categories)
        self.label_names = ["World", "Sports", "Business", "Sci/Tech"]
    
    def predict(self, text):
        if not text.strip():
            return {
                "prediction": "Unknown",
                "confidence": 0.0,
                "probabilities": {label: 0.0 for label in self.label_names}
            }
        
        # Tokenize input text
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        
        # Move inputs to the same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        return {
            "prediction": self.label_names[predicted_class],
            "confidence": confidence,
            "probabilities": {
                label: prob.item() for label, prob in zip(self.label_names, probabilities[0])
            }
        }

def evaluate_model(classifier, test_dataset, sample_size=1000):
    """Evaluate model on test dataset"""
    texts = test_dataset["text"][:sample_size]
    true_labels = test_dataset["label"][:sample_size]
    
    predictions = []
    confidences = []
    
    print(f"Evaluating on {len(texts)} samples...")
    
    for i, text in enumerate(texts):
        if i % 100 == 0:
            print(f"Processed {i}/{len(texts)} samples")
        result = classifier.predict(text)
        predictions.append(result["prediction"])
        confidences.append(result["confidence"])
    
    # Convert predictions to numerical labels
    pred_numeric = [classifier.label_names.index(pred) for pred in predictions]
    
    # Classification report
    print("\n" + "="*50)
    print("Classification Report:")
    print("="*50)
    print(classification_report(true_labels, pred_numeric, 
                              target_names=classifier.label_names, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_numeric)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classifier.label_names,
                yticklabels=classifier.label_names)
    plt.title('Confusion Matrix', fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Confidence statistics
    print("\n" + "="*50)
    print("Confidence Statistics:")
    print("="*50)
    print(f"Average Confidence: {np.mean(confidences):.3f}")
    print(f"Confidence Std: {np.std(confidences):.3f}")
    print(f"Min Confidence: {np.min(confidences):.3f}")
    print(f"Max Confidence: {np.max(confidences):.3f}")
    
    # Accuracy by class
    accuracy_by_class = []
    for i, label in enumerate(classifier.label_names):
        mask = np.array(true_labels) == i
        if np.sum(mask) > 0:
            class_acc = np.mean(np.array(pred_numeric)[mask] == np.array(true_labels)[mask])
            accuracy_by_class.append((label, class_acc))
    
    print("\nAccuracy by Class:")
    for label, acc in accuracy_by_class:
        print(f"{label}: {acc:.3f}")
    
    return pred_numeric, true_labels, confidences

def quick_test(classifier):
    """Quick test with sample headlines"""
    test_headlines = [
        "Stock market reaches all-time high",
        "Football team wins championship",
        "New scientific discovery in quantum computing",
        "International peace talks continue"
    ]
    
    print("Quick Test Results:")
    print("="*40)
    for headline in test_headlines:
        result = classifier.predict(headline)
        print(f"Headline: {headline}")
        print(f"Predicted: {result['prediction']} (Confidence: {result['confidence']:.3f})")
        print("-" * 40)

# Usage
if __name__ == "__main__":
    try:
        # Load classifier
        print("Loading classifier...")
        classifier = NewsClassifier()
        
        # Quick test
        quick_test(classifier)
        
        # Load test data
        print("\nLoading test dataset...")
        dataset = load_dataset("ag_news")
        test_data = dataset["test"]
        
        # Evaluate
        evaluate_model(classifier, test_data, sample_size=500)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease train the model first by running:")
        print("python fine_tune_bert.py")
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
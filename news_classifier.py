# news_classifier.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

class NewsClassifier:
    def __init__(self, model_path="./bert-ag-news-final"):
        """
        Initialize the news classifier with a fine-tuned BERT model
        
        Args:
            model_path (str): Path to the fine-tuned model directory
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Convert relative path to absolute path
        model_path = os.path.abspath(model_path)
        
        # Check if model exists locally
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        # Load tokenizer and model from local directory
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Label mapping (AG News categories)
        self.label_names = ["World", "Sports", "Business", "Sci/Tech"]
    
    def predict(self, text):
        """
        Predict the topic of a news headline
        
        Args:
            text (str): News headline to classify
            
        Returns:
            dict: Prediction results with confidence scores
        """
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
    
    def batch_predict(self, texts):
        """
        Predict topics for multiple texts
        
        Args:
            texts (list): List of news headlines
            
        Returns:
            list: List of prediction results
        """
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results
# app.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import numpy as np
import os

class NewsClassifier:
    def __init__(self, model_path="./bert-ag-news-final"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Convert to absolute path and ensure it's treated as local
        model_path = os.path.abspath(model_path)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()
        
        self.label_names = ["World", "Sports", "Business", "Sci/Tech"]
    
    def predict(self, text):
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
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

def main():
    st.set_page_config(
        page_title="News Topic Classifier",
        page_icon="📰",
        layout="wide"
    )
    
    st.title("📰 News Topic Classifier")
    st.markdown("Classify news headlines into categories using fine-tuned BERT")
    
    # Initialize classifier
    @st.cache_resource
    def load_classifier():
        return NewsClassifier()
    
    try:
        classifier = load_classifier()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please make sure you've trained the model first by running the training script.")
        return
    
    # Rest of your Streamlit app code...
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter News Headline")
        headline = st.text_area(
            "Paste your news headline here:",
            height=100,
            placeholder="e.g., 'Stock market reaches all-time high record'"
        )
        
        if st.button("Classify", type="primary"):
            if headline.strip():
                with st.spinner("Analyzing..."):
                    result = classifier.predict(headline)
                
                st.success(f"**Predicted Topic:** {result['prediction']}")
                st.info(f"**Confidence:** {result['confidence']:.2%}")
                
                # Display probabilities
                st.subheader("Probability Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                labels = list(result['probabilities'].keys())
                probabilities = list(result['probabilities'].values())
                
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                bars = ax.bar(labels, probabilities, color=colors)
                
                ax.set_ylabel('Probability')
                ax.set_title('Topic Probabilities')
                ax.set_ylim(0, 1)
                
                for bar, prob in zip(bars, probabilities):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{prob:.2%}', ha='center', va='bottom')
                
                st.pyplot(fig)
                
            else:
                st.warning("Please enter a news headline.")
    
    with col2:
        st.subheader("About")
        st.markdown("""
        This classifier uses BERT fine-tuned on the AG News dataset to categorize news headlines into:
        
        - **World**: International news, politics
        - **Sports**: Athletic events, games
        - **Business**: Finance, companies, economy
        - **Sci/Tech**: Science, technology, innovation
        
        **Model**: bert-base-uncased  
        **Dataset**: AG News (120,000 training samples)
        """)
        
        st.subheader("Example Headlines")
        examples = [
            "Stock market reaches all-time high → Business",
            "Football team wins championship → Sports",
            "New quantum computing breakthrough → Sci/Tech",
            "International peace talks continue → World"
        ]
        
        for example in examples:
            st.caption(f"• {example}")

if __name__ == "__main__":
    main()
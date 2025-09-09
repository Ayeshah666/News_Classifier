## News Topic Classification with BERT
A complete pipeline for fine-tuning BERT on the AG News dataset to classify news headlines into categories, with deployment options using Streamlit and Gradio.

## 📋 Project Overview
This project demonstrates:

Fine-tuning BERT-base-uncased on the AG News dataset

News topic classification into 4 categories: World, Sports, Business, Sci/Tech

Model evaluation with comprehensive metrics

Web deployment using Streamlit and Gradio

Safety features and error handling

🏗️ Project Structure
text
news-classification/
├── fine_tune_bert.py          # BERT fine-tuning script
├── manual_train.py            # Manual training alternative
├── news_classifier.py         # Inference class
├── app.py                     # Streamlit web app
├── evaluate.py                # Model evaluation script
├── README.md                  # This file
└── bert_ag_news_final/        # Fine-tuned model (after training)
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer_config.json
    └── vocab.txt
## 🚀 Quick Start
1. Installation
bash
# Clone repository
git clone https://github.com/your-username/news-classification.git
cd news-classification

# Create virtual environment
python -m venv news_env
source news_env/bin/activate  # On Windows: news_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
2. Fine-tune BERT
bash
# Option 1: Using Trainer (if supported)
python fine_tune_bert.py

# Option 2: Manual training (recommended for older versions)
python manual_train.py
3. Evaluate Model
bash
python evaluate.py
4. Deploy Web App
bash
# Streamlit app
streamlit run app.py

# Gradio app
python gradio_app.py
## 📊 Dataset
The project uses the AG News Dataset from Hugging Face:

120,000 training samples

7,600 test samples

4 categories: World, Sports, Business, Sci/Tech

## 🎯 Model Performance
After fine-tuning, expect:

Accuracy: 92-94%

F1-score: 92-94% (weighted)

Inference time: <100ms per headline

## 🔧 Technical Details
Model Architecture
Base Model: bert-base-uncased

Task: Sequence Classification

Output: 4-class probabilities

Training Parameters
Learning Rate: 2e-5

Batch Size: 16

Epochs: 2-3

Max Sequence Length: 128 tokens

## Safety Features
Input validation

Harmful query filtering

Confidence scoring

Error handling

## 🌐 Web Deployment
Streamlit App Features
Real-time headline classification

Probability visualization

Example headlines

Responsive design

Gradio App Features
Simple interface

Batch processing support

API-like functionality

## 📈 Evaluation Metrics
The evaluation script provides:

Classification report (precision, recall, F1)

Confusion matrix visualization

Confidence statistics

Per-class accuracy analysis

## 🛠️ Customization
Adding New Categories
Update label mapping in training scripts

Retrain model with new dataset

Update inference class labels

## Model Optimization
Adjust hyperparameters in fine_tune_bert.py

Modify batch size based on GPU memory

Experiment with different learning rates


Streamlit and Gradio teams for deployment frameworks

📞 Support
If you have any questions or issues:

Check the Issues page

Create a new issue with detailed description

Provide your environment details and error logs

🎯 Future Enhancements
Docker containerization

API deployment with FastAPI

Model quantization for faster inference

Multi-language support

Real-time training monitoring

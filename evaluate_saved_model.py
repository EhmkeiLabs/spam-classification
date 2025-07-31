import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """Preprocess the text data in the same way as during training"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]", " ", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics"""
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Print classification report
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], 
                yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    # Print key metrics
    print("\n=== MODEL PERFORMANCE METRICS ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

def main():
    # Load the saved model
    model_path = 'spam_detection_model.joblib'
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # Load your test data (replace with your actual test data loading code)
    # For demonstration, we'll use the same data loading as in the notebook
    # You should replace this with your actual test data
    print("Loading and preprocessing test data...")
    
    try:
        # Load the SMS Spam Collection data
        df_test = pd.read_csv('sms_spam_collection/SMSSpamCollection', 
                            sep='\t', 
                            header=None, 
                            names=['label', 'message'])
        print("Test data loaded successfully.")
        
        # Preprocess the test data (same as training preprocessing)
        df_test['message_processed'] = df_test['message'].apply(preprocess_text)
        
        # Prepare features and labels
        X_test = df_test['message_processed']
        y_test = df_test['label'].apply(lambda x: 1 if x == "spam" else 0)
        
        # Evaluate the model
        print("\nEvaluating model performance...")
        metrics = evaluate_model(model, X_test, y_test)
        
    except Exception as e:
        print(f"Error loading or processing test data: {e}")
        print("\nTo use this script, please ensure you have a test dataset")
        print("or modify the script to load your test data.")

if __name__ == "__main__":
    main()

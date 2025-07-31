# Model Evaluation
# Add this code to a new cell in your Jupyter notebook

from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Function to evaluate model
def evaluate_model(model, X, y_true, cv=5):
    """
    Evaluate the model using cross-validation and return metrics and plots
    """
    # Get cross-validated predictions
    y_pred = cross_val_predict(model, X, y_true, cv=cv)
    y_proba = cross_val_predict(model, X, y_true, cv=cv, method='predict_proba') 
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Print classification report
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_true, y_pred, target_names=['Ham', 'Spam']))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], 
                yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
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
    
    # Plot Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba[:, 1])
    avg_precision = average_precision_score(y_true, y_proba[:, 1])
    
    plt.figure(figsize=(8, 6))
    plt.step(recall_curve, precision_curve, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve (AP={0:0.2f})'.format(avg_precision))
    plt.show()
    
    # Feature importance (for the final model)
    try:
        if hasattr(model.named_steps['classifier'], 'feature_log_prob_'):
            feature_names = model.named_steps['vectorizer'].get_feature_names_out()
            log_prob = model.named_steps['classifier'].feature_log_prob_
            
            # Get top 20 important features for spam class
            top_spam_features = np.argsort(log_prob[1])[-20:]
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_spam_features)), log_prob[1][top_spam_features], align='center')
            plt.yticks(range(len(top_spam_features)), [feature_names[i] for i in top_spam_features])
            plt.xlabel('Log Probability')
            plt.title('Top 20 Important Features for Spam Detection')
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"Could not plot feature importance: {e}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision
    }

# Example usage (uncomment and run in your notebook):
"""
# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], y, test_size=0.2, random_state=42, stratify=y
)

# Train the best model on the full training set
best_model.fit(X_train, y_train)

# Evaluate on test set
print("\n=== FINAL MODEL EVALUATION ===")
evaluation_results = evaluate_model(best_model, X_test, y_test)
"""

# To evaluate using cross-validation on the entire dataset:
# evaluation_results = evaluate_model(best_model, df["message"], y)

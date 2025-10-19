# evaluators.py
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def evaluate(self, model, test_dataset):
        """Evaluate model on test dataset"""
        y_true = []
        y_pred = []
        y_prob = []
        
        for x_batch, y_batch in test_dataset:
            predictions = model(x_batch, training=False)
            y_true.extend(y_batch.numpy())
            y_pred.extend(tf.argmax(predictions, axis=1).numpy())
            y_prob.extend(predictions.numpy())
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # ROC-AUC (one-vs-rest for multi-class)
        try:
            roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except:
            roc_auc = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
    
    def plot_training_history(self, history, model_name):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Val Loss')
        ax1.set_title(f'{model_name} - Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy plot
        ax2.plot(history['train_acc'], label='Train Acc')
        ax2.plot(history['val_acc'], label='Val Acc')
        ax2.set_title(f'{model_name} - Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

# utils.py
import tensorflow as tf
import numpy as np
import os

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def compute_model_complexity(model):
    """Compute number of parameters and FLOPs"""
    trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    
    return {
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'total_params': trainable_params + non_trainable_params
    }

def save_model(model, filepath):
    """Save model weights"""
    model.save_weights(filepath)

def load_model(model, filepath):
    """Load model weights"""
    model.load_weights(filepath)
    return model

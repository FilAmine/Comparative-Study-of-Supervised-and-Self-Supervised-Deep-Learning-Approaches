import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os

class CICIoT2023DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        
    def load_and_preprocess(self):
        """Load and preprocess the CICIoT2023 dataset"""
        # Load dataset (assuming CSV format)
        df = pd.read_csv(self.data_path)
        
        # Remove duplicates and missing values
        df = df.drop_duplicates().dropna()
        
        # Separate features and labels
        features = df.drop('label', axis=1)
        labels = df['label']
        
        # Normalize numerical features
        numerical_cols = features.select_dtypes(include=[np.number]).columns
        features[numerical_cols] = self.scaler.fit_transform(features[numerical_cols])
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        return features.values, encoded_labels, self.label_encoder.classes_
    
    def prepare_datasets(self, label_fraction=1.0, batch_size=128, zero_day_families=10):
        """Prepare datasets for training and evaluation"""
        X, y, classes = self.load_and_preprocess()
        
        # Initial train/test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.15/0.85, random_state=42, stratify=y_temp
        )
        
        # Simulate zero-day attacks by holding out some families
        if zero_day_families > 0:
            unique_classes = np.unique(y_train)
            held_out_classes = np.random.choice(
                unique_classes, 
                size=zero_day_families, 
                replace=False
            )
            
            # Filter test set to include only held-out classes for zero-day evaluation
            zero_day_mask = np.isin(y_test, held_out_classes)
            X_zero_day = X_test[zero_day_mask]
            y_zero_day = y_test[zero_day_mask]
            
            # Regular test set (seen classes)
            X_test = X_test[~zero_day_mask]
            y_test = y_test[~zero_day_mask]
        
        # Create TensorFlow datasets
        def create_dataset(X, y, labeled=True, training=False):
            dataset = tf.data.Dataset.from_tensor_slices((X, y))
            if training:
                dataset = dataset.shuffle(10000)
            if labeled:
                dataset = dataset.batch(batch_size)
            else:
                # For SSL, we don't need labels during pretraining
                dataset = dataset.map(lambda x, y: (x,))  # Remove labels
                dataset = dataset.batch(batch_size)
            return dataset.prefetch(tf.data.AUTOTUNE)
        
        # Apply label fraction for scarcity experiments
        if label_fraction < 1.0:
            X_train_labeled, _, y_train_labeled, _ = train_test_split(
                X_train, y_train, 
                train_size=label_fraction, 
                random_state=42, 
                stratify=y_train
            )
        else:
            X_train_labeled, y_train_labeled = X_train, y_train
        
        datasets = {
            'train_labeled': create_dataset(X_train_labeled, y_train_labeled, training=True),
            'train_unlabeled': create_dataset(X_train, y_train, labeled=False, training=True),
            'val': create_dataset(X_val, y_val),
            'test': create_dataset(X_test, y_test),
        }
        
        if zero_day_families > 0:
            datasets['zero_day'] = create_dataset(X_zero_day, y_zero_day)
        
        return datasets

class DataAugmentation:
    """Data augmentation for SimCLR"""
    
    @staticmethod
    def random_scaling(x, scale_range=0.1):
        """Random scaling augmentation"""
        scale_factor = tf.random.uniform([], 1-scale_range, 1+scale_range)
        return x * scale_factor
    
    @staticmethod
    def gaussian_noise(x, std=0.01):
        """Add Gaussian noise"""
        noise = tf.random.normal(tf.shape(x), stddev=std)
        return x + noise
    
    @staticmethod
    def random_masking(x, mask_prob=0.1):
        """Random feature masking"""
        mask = tf.random.uniform(tf.shape(x)) > mask_prob
        return x * tf.cast(mask, tf.float32)
    
    @staticmethod
    def simclr_augment(x):
        """Apply SimCLR augmentations to create two views"""
        view1 = x
        view2 = x
        
        # Apply augmentation pipeline
        for aug in [DataAugmentation.random_scaling, 
                   DataAugmentation.gaussian_noise,
                   DataAugmentation.random_masking]:
            view1 = aug(view1)
            view2 = aug(view2)
        
        return view1, view2

import tensorflow as tf
import numpy as np
from data_loader import DataAugmentation

class CNNTrainer:
    """Trainer for supervised CNN model"""
    
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        self.train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        self.val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    
    @tf.function
    def train_step(self, x_batch, y_batch):
        with tf.GradientTape() as tape:
            predictions = self.model(x_batch, training=True)
            loss = self.loss_fn(y_batch, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_acc_metric.update_state(y_batch, predictions)
        return loss
    
    @tf.function
    def val_step(self, x_batch, y_batch):
        predictions = self.model(x_batch, training=False)
        loss = self.loss_fn(y_batch, predictions)
        self.val_acc_metric.update_state(y_batch, predictions)
        return loss
    
    def train(self, train_dataset, val_dataset, epochs=100):
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Training
            train_losses = []
            for x_batch, y_batch in train_dataset:
                loss = self.train_step(x_batch, y_batch)
                train_losses.append(loss)
            
            # Validation
            val_losses = []
            for x_batch, y_batch in val_dataset:
                loss = self.val_step(x_batch, y_batch)
                val_losses.append(loss)
            
            # Record metrics
            train_loss = np.mean(train_losses)
            train_acc = self.train_acc_metric.result()
            val_loss = np.mean(val_losses)
            val_acc = self.val_acc_metric.result()
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Reset metrics
            self.train_acc_metric.reset_states()
            self.val_acc_metric.reset_states()
        
        return history

class SimCLRTrainer:
    """Trainer for SimCLR model"""
    
    def __init__(self, model, learning_rate=0.3):
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        
        # Metrics
        self.pretrain_loss_metric = tf.keras.metrics.Mean()
        self.train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        self.val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    
    @tf.function
    def pretrain_step(self, x_batch):
        # Generate two augmented views
        x_view1, x_view2 = DataAugmentation.simclr_augment(x_batch)
        
        with tf.GradientTape() as tape:
            # Get projections for both views
            projections1 = self.model.project(x_view1, training=True)
            projections2 = self.model.project(x_view2, training=True)
            
            # Compute contrastive loss
            loss = self.model.contrastive_loss(projections1, projections2)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.pretrain_loss_metric.update_state(loss)
        return loss
    
    def pretrain(self, unlabeled_dataset, epochs=500):
        """Self-supervised pretraining"""
        history = {'pretrain_loss': []}
        
        for epoch in range(epochs):
            for x_batch in unlabeled_dataset:
                loss = self.pretrain_step(x_batch)
            
            pretrain_loss = self.pretrain_loss_metric.result()
            history['pretrain_loss'].append(pretrain_loss)
            
            if epoch % 50 == 0:
                print(f'Pretrain Epoch {epoch+1}/{epochs}: Loss: {pretrain_loss:.4f}')
            
            self.pretrain_loss_metric.reset_states()
        
        return history
    
    def finetune(self, train_dataset, val_dataset, epochs=50, fine_tune_lr=0.001):
        """Supervised fine-tuning"""
        # Use smaller learning rate for fine-tuning
        fine_tune_optimizer = tf.keras.optimizers.Adam(learning_rate=fine_tune_lr)
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Training
            train_losses = []
            for x_batch, y_batch in train_dataset:
                with tf.GradientTape() as tape:
                    predictions = self.model(x_batch, training=True)
                    loss = self.loss_fn(y_batch, predictions)
                
                # Only update classification head (optional: can also update encoder)
                trainable_vars = self.model.classification_head.trainable_variables
                gradients = tape.gradient(loss, trainable_vars)
                fine_tune_optimizer.apply_gradients(zip(gradients, trainable_vars))
                
                self.train_acc_metric.update_state(y_batch, predictions)
                train_losses.append(loss)
            
            # Validation
            val_losses = []
            for x_batch, y_batch in val_dataset:
                predictions = self.model(x_batch, training=False)
                loss = self.loss_fn(y_batch, predictions)
                self.val_acc_metric.update_state(y_batch, predictions)
                val_losses.append(loss)
            
            # Record metrics
            train_loss = np.mean(train_losses)
            train_acc = self.train_acc_metric.result()
            val_loss = np.mean(val_losses)
            val_acc = self.val_acc_metric.result()
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f'Fine-tune Epoch {epoch+1}/{epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Reset metrics
            self.train_acc_metric.reset_states()
            self.val_acc_metric.reset_states()
        
        return history
    
    def get_classifier(self):
        """Get the model for classification"""
        return self.model

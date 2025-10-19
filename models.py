import tensorflow as tf
from tensorflow.keras import layers, Model

class CNNModel(Model):
    """Supervised CNN model for IoT malware detection"""
    
    def __init__(self, num_classes=74):
        super(CNNModel, self).__init__()
        
        self.conv1 = layers.Conv1D(64, 3, activation='relu', input_shape=(84, 1))
        self.pool1 = layers.MaxPooling1D(2)
        self.conv2 = layers.Conv1D(128, 3, activation='relu')
        self.pool2 = layers.MaxPooling1D(2)
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.classifier = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        # Reshape for 1D convolution
        x = tf.expand_dims(inputs, -1)
        
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.classifier(x)

class ResNet1DBlock(layers.Layer):
    """Residual block for 1D signals"""
    
    def __init__(self, filters, kernel_size=3, stride=1):
        super(ResNet1DBlock, self).__init__()
        
        self.conv1 = layers.Conv1D(filters, kernel_size, strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(filters, kernel_size, padding='same')
        self.bn2 = layers.BatchNormalization()
        
        self.shortcut = layers.Conv1D(filters, 1, strides=stride) if stride != 1 else lambda x: x
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        shortcut = self.shortcut(inputs)
        x = layers.add([x, shortcut])
        return tf.nn.relu(x)

class SimCLRModel(Model):
    """SimCLR model for self-supervised learning"""
    
    def __init__(self, input_dim=84, projection_dim=128, temperature=0.1):
        super(SimCLRModel, self).__init__()
        self.temperature = temperature
        
        # Encoder (Adapted ResNet-18 for 1D)
        self.encoder = self._build_encoder(input_dim)
        
        # Projection head
        self.projection_head = tf.keras.Sequential([
            layers.Dense(2048, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(projection_dim)
        ])
        
        # Classification head (for fine-tuning)
        self.classification_head = layers.Dense(74, activation='softmax')
    
    def _build_encoder(self, input_dim):
        """Build 1D ResNet-18 like encoder"""
        inputs = tf.keras.Input(shape=(input_dim, 1))
        
        # Initial conv
        x = layers.Conv1D(64, 7, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        x = layers.MaxPooling1D(3, strides=2, padding='same')(x)
        
        # Residual blocks
        x = ResNet1DBlock(64)(x)
        x = ResNet1DBlock(64)(x)
        
        x = ResNet1DBlock(128, stride=2)(x)
        x = ResNet1DBlock(128)(x)
        
        x = ResNet1DBlock(256, stride=2)(x)
        x = ResNet1DBlock(256)(x)
        
        x = ResNet1DBlock(512, stride=2)(x)
        x = ResNet1DBlock(512)(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        return tf.keras.Model(inputs, x, name='encoder')
    
    def encode(self, inputs, training=False):
        """Get representations from encoder"""
        # Add channel dimension if needed
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, -1)
        return self.encoder(inputs, training=training)
    
    def project(self, inputs, training=False):
        """Project to contrastive space"""
        representations = self.encode(inputs, training=training)
        return self.projection_head(representations, training=training)
    
    def call(self, inputs, training=False):
        """Forward pass for classification"""
        representations = self.encode(inputs, training=training)
        return self.classification_head(representations)
    
    def contrastive_loss(self, projections1, projections2):
        """Compute NT-Xent loss"""
        # Normalize projections
        projections1 = tf.math.l2_normalize(projections1, axis=1)
        projections2 = tf.math.l2_normalize(projections2, axis=1)
        
        batch_size = tf.shape(projections1)[0]
        
        # Concatenate all projections
        all_projections = tf.concat([projections1, projections2], axis=0)
        
        # Compute similarity matrix
        similarity_matrix = tf.matmul(all_projections, all_projections, transpose_b=True)
        
        # Create labels for positive pairs
        labels = tf.range(batch_size)
        labels = tf.concat([labels + batch_size, labels], axis=0)
        
        # Mask for positive pairs
        mask = tf.one_hot(labels, 2 * batch_size)
        
        # Subtract max for numerical stability
        logits = similarity_matrix / self.temperature
        logits_max = tf.reduce_max(logits, axis=1, keepdims=True)
        logits = logits - logits_max
        
        # Compute loss
        exp_logits = tf.exp(logits)
        log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits * (1 - mask), axis=1, keepdims=True))
        
        loss = -tf.reduce_sum(log_prob * mask) / (2 * batch_size)
        return loss

# Experiment configuration
class Config:
    # Data parameters
    DATA_PATH = "data/CICIoT2023.csv"
    BATCH_SIZE = 128
    LABEL_FRACTIONS = [1.0, 0.5, 0.3, 0.1]  # For label scarcity experiments
    
    # Model parameters
    INPUT_DIM = 84
    NUM_CLASSES = 74
    PROJECTION_DIM = 128
    TEMPERATURE = 0.1
    
    # Training parameters
    PRETRAIN_EPOCHS = 500
    FINETUNE_EPOCHS = 50
    SUPERVISED_EPOCHS = 100
    
    # Optimization
    PRETRAIN_LR = 0.3
    FINETUNE_LR = 0.001
    SUPERVISED_LR = 0.001
    
    # Experimental setup
    ZERO_DAY_FAMILIES = 10
    CROSS_VALIDATION_FOLDS = 5
    RANDOM_SEED = 42

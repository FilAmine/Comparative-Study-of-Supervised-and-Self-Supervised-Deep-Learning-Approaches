import argparse
import numpy as np
import tensorflow as tf
from data_loader import CICIoT2023DataLoader
from models import CNNModel, SimCLRModel
from trainers import CNNTrainer, SimCLRTrainer
from evaluators import ModelEvaluator
import utils

def main():
    parser = argparse.ArgumentParser(description='IoT Malware Detection')
    parser.add_argument('--model', type=str, required=True, choices=['cnn', 'simclr'])
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--label_fraction', type=float, default=1.0)
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    utils.set_random_seeds(42)
    
    # Load and preprocess data
    print("Loading CICIoT2023 dataset...")
    data_loader = CICIoT2023DataLoader(args.data_path)
    datasets = data_loader.prepare_datasets(
        label_fraction=args.label_fraction,
        batch_size=args.batch_size
    )
    
    if args.model == 'cnn':
        # CNN Training
        model = CNNModel(num_classes=74)
        trainer = CNNTrainer(model, learning_rate=args.learning_rate)
        history = trainer.train(
            datasets['train_labeled'],
            datasets['val'],
            epochs=args.epochs
        )
        
    else:  # simclr
        # SimCLR Training
        model = SimCLRModel(
            input_dim=84,
            projection_dim=128,
            temperature=0.1
        )
        trainer = SimCLRTrainer(model, learning_rate=0.3)  # Higher LR for SSL
        
        # Pretraining
        print("Starting SimCLR pretraining...")
        pretrain_history = trainer.pretrain(
            datasets['train_unlabeled'],
            epochs=500
        )
        
        # Fine-tuning
        print("Starting fine-tuning...")
        finetune_history = trainer.finetune(
            datasets['train_labeled'],
            datasets['val'],
            epochs=50
        )
    
    # Evaluation
    print("Evaluating model...")
    evaluator = ModelEvaluator()
    results = evaluator.evaluate(
        trainer.get_classifier() if args.model == 'simclr' else model,
        datasets['test']
    )
    
    print("\n=== Final Results ===")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()

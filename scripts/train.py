#!/usr/bin/env python3
"""
Main training script for MultiFuse models.

Usage:
    python scripts/train.py --config configs/experiments/multifuse_v1.yaml
    python scripts/train.py --config configs/experiments/efficientnet_baseline.yaml
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import torchvision.transforms as transforms
import torchvision.models as models
import yaml
from src.training.trainer import train
from src.data.loaders import create_dataloaders
from src.utils.config import load_config
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train MultiFuse models")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for models and logs"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1
    
    # Setup logging
    setup_logging(config.get('logging', {}))
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and config.get('device', {}).get('use_cuda', True) else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    
    try:
        # Create dataloaders based on config
        training_config = config.get('training', {})
        data_config = config.get('data', {})
        
        # For now, use default paths if not specified
        training_dir = data_config.get('training_directory', 'data/processed')
        test_dir = data_config.get('test_directory', 'data/processed')
        batch_size = training_config.get('batch_size', 32)
        
        # Basic transform (can be enhanced later)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Creating dataloaders with:")
        print(f"  Training dir: {training_dir}")
        print(f"  Test dir: {test_dir}")
        print(f"  Batch size: {batch_size}")
        
        # For now, we'll skip the actual dataloader creation since the dataset paths may not exist
        # In a real scenario, you would uncomment the next line:
        # train_loader, test_loader, class_names = create_dataloaders(training_dir, test_dir, transform, batch_size)
        
        # Create a simple model for demonstration (EfficientNet baseline)
        model_config = config.get('model', {})
        num_classes = model_config.get('num_classes', 50)
        
        print(f"Creating model with {num_classes} classes")
        model = models.efficientnet_b0(pretrained=True)
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        )
        model = model.to(device)
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=training_config.get('learning_rate', 0.001))
        loss_fn = torch.nn.CrossEntropyLoss()
        
        print("Training setup complete!")
        print(f"Model: {model.__class__.__name__}")
        print(f"Optimizer: {optimizer.__class__.__name__}")
        print(f"Loss function: {loss_fn.__class__.__name__}")
        print(f"Device: {device}")
        
        # Note: Actual training would happen here with the train() function
        # For now, we just demonstrate that the setup works
        print("\nTo run actual training, ensure your dataset is properly set up and uncomment the training loop.")
        
    except Exception as e:
        print(f"Error during training setup: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

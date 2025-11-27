import os
import torch
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    """Base trainer interface for all models"""
    
    def __init__(self, model, device, output_dir):
        """
        Args:
            model: The model to train
            device: Device to use for training
            output_dir: Directory to save outputs
        """
        self.model = model
        self.device = device
        self.output_dir = output_dir
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
    
    @abstractmethod
    def train(self, dataloader, n_epochs, **kwargs):
        """Train the model
        
        Args:
            dataloader: DataLoader for training data
            n_epochs: Number of epochs to train
            **kwargs: Additional training parameters
        """
        pass
    
    def save_model(self, path):
        """Save model checkpoint
        
        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, path)
    
    def load_model(self, path):
        """Load model checkpoint
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return self.model 
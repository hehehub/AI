import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """Base model interface for all models (VAE, Diffusion, etc.)"""
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass of the model"""
        pass
    
    @abstractmethod
    def train_step(self, batch, optimizer):
        """Perform a single training step
        
        Args:
            batch: The batch data from dataloader
            optimizer: The optimizer for the model
            
        Returns:
            loss: The loss value for this batch
        """
        pass
    
    @abstractmethod
    def inference(self, *args, **kwargs):
        """Generate samples or reconstructions"""
        pass
    
    @abstractmethod
    def evaluate(self, dataloader):
        """Evaluate the model on a dataset
        
        Args:
            dataloader: DataLoader for evaluation dataset
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        pass 
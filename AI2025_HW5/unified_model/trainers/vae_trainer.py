import os
import torch
from tqdm import tqdm

from unified_model.trainers.base_trainer import BaseTrainer

class VAETrainer(BaseTrainer):
    """Trainer for VAE models"""
    
    def __init__(self, model, device, output_dir, model_name='vae.pth'):
        """
        Args:
            model: VAE model instance
            device: Device to use for training
            output_dir: Directory to save outputs
            model_name: Name of the model file
        """
        super().__init__(model, device, output_dir)
        self.model_name = model_name
        os.makedirs(output_dir, exist_ok=True)
    
    def train(self, train_loader, val_loader=None, n_epochs=50, beta=1.0, lr=2e-4, save_interval=10):
        """Train the VAE model
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            n_epochs: Number of epochs to train
            beta: Weight for KL divergence term
            lr: Learning rate
            save_interval: Interval to save model checkpoints
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        best_val_loss = float('inf')
        
        for epoch in range(n_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            n_batches = 0
            
            pbar = tqdm(total=len(train_loader.dataset))
            pbar.set_description(f'Train Epoch {epoch + 1}')
            
            for batch in train_loader:
                batch_size = batch[0].size(0)
                n_batches += batch_size
                
                # Train step
                loss = self.model.train_step(batch, optimizer)
                train_loss += loss * batch_size
                
                pbar.update(batch_size)
                pbar.set_description(f'Train Epoch {epoch + 1}, Loss: {train_loss / n_batches:.6f}')
            
            pbar.close()
            
            # Validation phase
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                val_loss = val_metrics['total_loss']
                print(f'Validation Loss: {val_loss:.6f}')
                
                # Save best model
                if val_loss < best_val_loss:
                    print(f'Validation loss improved from {best_val_loss:.6f} to {val_loss:.6f}')
                    best_val_loss = val_loss
                    self.save_model(os.path.join(self.output_dir, 'vae_best.pth'))
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                checkpoint_dir = os.path.join(self.output_dir, f'epoch_{epoch + 1}')
                os.makedirs(checkpoint_dir, exist_ok=True)
                self.save_model(os.path.join(checkpoint_dir, self.model_name))
                
                # Generate and save sample images
                label = torch.eye(10).repeat(10, 1).to(self.device)  # Assuming 10 classes
                self.model.sample_images(label, save=True, save_dir=checkpoint_dir)
    
    def evaluate(self, dataloader):
        """Evaluate the model
        
        Args:
            dataloader: DataLoader for evaluation
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        return self.model.evaluate(dataloader)
    
    def generate_dataset(self, n_samples_per_class=100, save_dir=None):
        """Generate a dataset of samples
        
        Args:
            n_samples_per_class: Number of samples per class
            save_dir: Directory to save generated samples
        """
        if save_dir is None:
            save_dir = os.path.join(self.output_dir, 'generated')
        
        self.model.make_dataset(n_samples_per_class=n_samples_per_class, save=True, save_dir=save_dir) 
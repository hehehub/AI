import os
import time
import torch
from tqdm import tqdm

from unified_model.trainers.base_trainer import BaseTrainer

class DiffusionTrainer(BaseTrainer):
    """Trainer for diffusion models"""
    
    def __init__(self, model, device, output_dir, model_name='model.pth'):
        """
        Args:
            model: DiffusionModel instance
            device: Device to use for training
            output_dir: Directory to save outputs
            model_name: Name of the model file
        """
        super().__init__(model, device, output_dir)
        self.model_name = model_name
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
    
    def train(self, dataloader, n_epochs, lr=1e-3, save_interval=10):
        """Train the diffusion model
        
        Args:
            dataloader: DataLoader for training data
            n_epochs: Number of epochs to train
            lr: Learning rate
            save_interval: Interval to save model checkpoints
        """
        optimizer = torch.optim.Adam(self.model.unet.parameters(), lr=lr)
        
        tic = time.time()
        for e in range(n_epochs):
            total_loss = 0
            n_batches = 0
            
            pbar = tqdm(total=len(dataloader.dataset))
            pbar.set_description('Train')
            
            for batch in dataloader:
                batch_size = batch[0].shape[0]
                n_batches += batch_size
                
                # Train step
                loss = self.model.train_step(batch, optimizer)
                total_loss += loss * batch_size
                
                pbar.update(batch_size)
                pbar.set_description(f'Epoch {e+1}, Loss: {total_loss / n_batches:.6f}')
            
            pbar.close()
            
            # Save model checkpoint
            if (e + 1) % save_interval == 0:
                checkpoint_dir = os.path.join(self.output_dir, f'epoch_{e+1}')
                os.makedirs(checkpoint_dir, exist_ok=True)
                self.save_model(os.path.join(checkpoint_dir, self.model_name))
                
            toc = time.time()
            print(f'Epoch {e+1} completed, avg loss {total_loss / n_batches:.6f}, time elapsed {(toc - tic):.2f}s')
    
    def save_model(self, path):
        """Save model checkpoint
        
        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.unet.state_dict(),
        }, path)
    
    def load_model(self, path):
        """Load model checkpoint
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path)
        self.model.unet.load_state_dict(checkpoint['model_state_dict'])
        return self.model 
import torch
import torch.nn as nn
import os
import torchvision

from unified_model.models.base_model import BaseModel
from unified_model.samplers.diffusion_samplers import DDPM
from unified_model.data.mnist_data import get_mnist_tensor_shape

class DiffusionModel(BaseModel):
    """Diffusion model implementation"""
    
    def __init__(self, unet, sampler_type='ddpm', n_steps=1000, device='cpu'):
        """
        Args:
            unet: UNet model for noise prediction
            sampler_type: Type of sampler to use ('ddpm')
            n_steps: Number of diffusion steps
            device: Device to use
        """
        super().__init__()
        self.unet = unet
        self.n_steps = n_steps
        self.device = device
        
        # Initialize sampler
        sampler_dict = {'ddpm': DDPM}
        self.sampler = sampler_dict[sampler_type](n_steps, device=device)
        self.mse_loss = nn.MSELoss()
        
    def forward(self, x, t):
        """Forward pass through the UNet
        
        Args:
            x: Input tensor
            t: Timestep tensor
            
        Returns:
            Predicted noise
        """
        return self.unet(x, t)
    
    def train_step(self, batch, optimizer):
        """Perform a single training step
        
        Args:
            batch: Batch data from dataloader (x, y)
            optimizer: Optimizer for the model
            
        Returns:
            loss: Loss value for this batch
        """
        x, _ = batch
        B = x.shape[0]
        x = x.to(self.device)
        
        # Sample timestep
        t = torch.randint(0, self.n_steps, (B, )).to(self.device)
        
        # Add noise
        eps = torch.randn_like(x).to(self.device)
        x_t = self.sampler.sample_forward(x, t, eps)
        
        # Predict noise
        eps_theta = self.unet(x_t, t.reshape(B, 1))
        
        # Calculate loss
        loss = self.mse_loss(eps_theta, eps)
        
        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def inference(self, shape):
        """Generate samples
        
        Args:
            shape: Shape of samples to generate (B, C, H, W)
            
        Returns:
            Generated samples
        """
        self.unet.eval()
        with torch.no_grad():
            samples = self.sampler.sample_backward(self.unet, shape).detach()
        return samples
    
    def evaluate(self, dataloader):
        """Evaluate model
        
        Args:
            dataloader: DataLoader for evaluation
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        # For diffusion models, we could implement FID or other metrics here
        # For simplicity, we just return an empty dictionary for now
        return {}
    
    @torch.no_grad()
    def sample_images(self, n_samples=64, save=True, save_dir='./diffusion'):
        """Sample images
        
        Args:
            n_samples: Number of samples to generate
            save: Whether to save images
            save_dir: Directory to save images
            
        Returns:
            Generated images
        """
        self.unet.eval()
        n_sample_per_side = int(n_samples ** 0.5)
        shape = (n_sample_per_side**2, *get_mnist_tensor_shape())
        samples = self.inference(shape)
        
        # Normalize from [-1, 1] to [0, 1] for saving
        imgs = ((samples + 1) / 2).clamp(0., 1.)
        
        if save:
            os.makedirs(save_dir, exist_ok=True)
            torchvision.utils.save_image(imgs, os.path.join(save_dir, 'sample.png'), nrow=n_sample_per_side)
        
        return imgs
    
    @torch.no_grad()
    def make_dataset(self, n_samples_per_class=10, save=True, save_dir='./diffusion/generated/'):
        """Generate dataset with samples for each class
        
        Args:
            n_samples_per_class: Number of samples per class
            save: Whether to save images
            save_dir: Directory to save images
            
        Returns:
            None
        """
        self.unet.eval()
        
        # Create directories for each class
        if save:
            for i in range(10):  # Assuming 10 classes for MNIST
                os.makedirs(os.path.join(save_dir, str(i)), exist_ok=True)
        
        # Generate samples for each "class" (just for organization)
        for i in range(10):
            # Generate samples
            shape = (n_samples_per_class, *get_mnist_tensor_shape())
            samples = self.inference(shape)
            
            # Normalize from [-1, 1] to [0, 1] for saving
            samples = ((samples + 1) / 2).clamp(0, 1)
            
            # Save individual images
            if save:
                for j in range(n_samples_per_class):
                    img = samples[j]
                    save_path = os.path.join(save_dir, str(i), f"{i}_{j:03d}.png")
                    torchvision.utils.save_image(img, save_path)

class ConditionalDiffusionModel(BaseModel):
    """Conditional diffusion model implementation with class guidance"""
    
    def __init__(self, conditional_unet, num_classes=10, sampler_type='ddpm', n_steps=1000, device='cpu'):
        """
        Args:
            conditional_unet: Conditional UNet model for noise prediction
            num_classes: Number of classes (default: 10 for MNIST)
            sampler_type: Type of sampler to use ('ddpm')
            n_steps: Number of diffusion steps
            device: Device to use
        """
        super().__init__()
        self.unet = conditional_unet
        self.n_steps = n_steps
        self.device = device
        self.num_classes = num_classes
        
        # Initialize sampler
        sampler_dict = {'ddpm': DDPM}
        self.sampler = sampler_dict[sampler_type](n_steps, device=device)
        self.mse_loss = nn.MSELoss()
        
    def forward(self, x, t, labels=None):
        """Forward pass through the conditional UNet
        
        Args:
            x: Input tensor
            t: Timestep tensor
            labels: Class labels (optional)
            
        Returns:
            Predicted noise
        """
        return self.unet(x, t, labels)
    
    def train_step(self, batch, optimizer):
        """Perform a single training step with class conditioning
        
        Args:
            batch: Batch data from dataloader (x, y)
            optimizer: Optimizer for the model
            
        Returns:
            loss: Loss value for this batch
        """
        x, y = batch
        B = x.shape[0]
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Sample timestep
        t = torch.randint(0, self.n_steps, (B, )).to(self.device)
        
        # Add noise
        eps = torch.randn_like(x).to(self.device)
        x_t = self.sampler.sample_forward(x, t, eps)
        
        # Predict noise with class conditioning
        eps_theta = self.unet(x_t, t, y)
        
        # Calculate loss
        loss = self.mse_loss(eps_theta, eps)
        
        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def inference(self, shape, labels=None):
        """Generate samples with class conditioning
        
        Args:
            shape: Shape of samples to generate (B, C, H, W)
            labels: Class labels (optional)
            
        Returns:
            Generated samples
        """
        self.unet.eval()
        
        def denoise_fn(x, t):
            # Class-conditioned denoising function
            if labels is not None:
                if len(t.shape) == 1:
                    return self.unet(x, t.unsqueeze(1), labels)
                else:
                    return self.unet(x, t, labels)
            else:
                if len(t.shape) == 1:
                    return self.unet(x, t.unsqueeze(1))
                else:
                    return self.unet(x, t)
        
        with torch.no_grad():
            samples = self.sampler.sample_backward(denoise_fn, shape).detach()
            
        return samples
    
    def evaluate(self, dataloader):
        """Evaluate model
        
        Args:
            dataloader: DataLoader for evaluation
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        # For diffusion models, we could implement FID or other metrics here
        # For simplicity, we just return an empty dictionary for now
        return {}
    
    @torch.no_grad()
    def sample_images(self, n_samples=64, save=True, save_dir='./diffusion'):
        """Sample images from random classes
        
        Args:
            n_samples: Number of samples to generate
            save: Whether to save images
            save_dir: Directory to save images
            
        Returns:
            Generated images
        """
        self.unet.eval()
        n_sample_per_side = int(n_samples ** 0.5)
        shape = (n_sample_per_side**2, *get_mnist_tensor_shape())
        
        # Generate labels - same label for each row
        labels = torch.zeros(shape[0], dtype=torch.long, device=self.device)
        for i in range(n_sample_per_side):
            # 为每一行分配相同的标签
            row_label = i % self.num_classes
            labels[i * n_sample_per_side:(i + 1) * n_sample_per_side] = row_label
        
        # Generate samples with labels
        samples = self.inference(shape, labels)
        
        # Normalize from [-1, 1] to [0, 1] for saving
        imgs = ((samples + 1) / 2).clamp(0., 1.)
        
        if save:
            os.makedirs(save_dir, exist_ok=True)
            torchvision.utils.save_image(imgs, os.path.join(save_dir, 'sample.png'), nrow=n_sample_per_side)
        return imgs
    
    @torch.no_grad()
    def sample_class_images(self, class_idx, n_samples=10, save=True, save_dir='./diffusion'):
        """Sample images for a specific class
        
        Args:
            class_idx: Class index to generate
            n_samples: Number of samples to generate
            save: Whether to save images
            save_dir: Directory to save images
            
        Returns:
            Generated images
        """
        self.unet.eval()
        
        # Create shape and labels
        shape = (n_samples, *get_mnist_tensor_shape())
        labels = torch.full((n_samples,), class_idx, dtype=torch.long, device=self.device)
        
        # Generate samples
        samples = self.inference(shape, labels)
        
        # Normalize from [-1, 1] to [0, 1] for saving
        imgs = ((samples + 1) / 2).clamp(0., 1.)
        
        if save:
            os.makedirs(save_dir, exist_ok=True)
            torchvision.utils.save_image(imgs, os.path.join(save_dir, f'class_{class_idx}.png'), nrow=int(n_samples**0.5))
        
        return imgs
    
    @torch.no_grad()
    def make_dataset(self, n_samples_per_class=10, save=True, save_dir='./diffusion/generated/'):
        """Generate dataset with samples for each class using class conditioning
        
        Args:
            n_samples_per_class: Number of samples per class
            save: Whether to save images
            save_dir: Directory to save images
            
        Returns:
            None
        """
        self.unet.eval()
        
        # Create directories for each class
        if save:
            for i in range(self.num_classes):
                os.makedirs(os.path.join(save_dir, str(i)), exist_ok=True)
        
        # Generate samples for each class
        for i in range(self.num_classes):
            # Create shape and labels for this class
            shape = (n_samples_per_class, *get_mnist_tensor_shape())
            labels = torch.full((n_samples_per_class,), i, dtype=torch.long, device=self.device)
            
            # Generate samples for this class
            samples = self.inference(shape, labels)
            
            # Normalize from [-1, 1] to [0, 1] for saving
            samples = ((samples + 1) / 2).clamp(0, 1)
            
            # Save individual images
            if save:
                for j in range(n_samples_per_class):
                    img = samples[j]
                    save_path = os.path.join(save_dir, str(i), f"{i}_{j:03d}.png")
                    torchvision.utils.save_image(img, save_path) 
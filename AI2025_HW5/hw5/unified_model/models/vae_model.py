import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from unified_model.models.base_model import BaseModel

class CVAE(BaseModel):
    """Conditional Variational Autoencoder implementation"""
    
    def __init__(self, img_size, label_size, latent_size, hidden_size, device='cpu'):
        """
        Args:
            img_size: Size of input image (C, H, W)
            label_size: Size of label (number of classes)
            latent_size: Size of latent space (z)
            hidden_size: Size of hidden layers (for linear layers, mlp, etc.)
            device: Device to use
        """
        super().__init__()
        self.img_size = img_size  # (C, H, W)
        self.img_dim = img_size[0] * img_size[1] * img_size[2]  # C*H*W
        self.label_size = label_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.device = device
        
        # Encoder
        # img -> fc ->                          -> fc -> mean   -> 
        #                concat -> encoder(MLP)                     z
        # label -> fc ->                        -> fc -> logstd -> 
        #########################################################
        #                TODO: VAE Encoder Architecture         #
        #########################################################
        # Hints:
        # 1. We need two seperate layers, one for image and one for label, which can be both linear layers.
        # 2. The output of the two layers could be concatenated and then fed into the encoder.
        # 3. The encoder can be a MLP
        # 4. The mean and logstd could be two seperate linear layers taking the output of the encoder as input.
        #########################################################
        self.img_encoder = nn.Linear(self.img_dim, hidden_size)
        self.label_encoder = nn.Linear(label_size, hidden_size)

        # 编码器：对拼接后的结果进行进一步处理
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # 均值与对数方差
        self.fc_mean = nn.Linear(hidden_size, latent_size)
        self.fc_logstd = nn.Linear(hidden_size, latent_size)
        #########################################################
        #                     End of TODO                       #
        #########################################################

        # Decoder
        # z -> fc ->
        #                concat -> decoder -> reconstruction
        # label -> fc ->
        #########################################################
        #                TODO: VAE Decoder Architecture         #
        #########################################################
        # Hints:
        # 1. We need two seperate layers, one for latent z and one for label, which can be both linear layers.
        # 2. The output of the two layers could be concatenated and then fed into the decoder.
        # 3. The decoder can be a MLP
        # 4. The decoder should have an output size of the image size.
        #########################################################
        self.z_decoder = nn.Linear(latent_size, hidden_size)
        self.label_decoder = nn.Linear(label_size, hidden_size)

        # 解码器：将拼接后的向量解码为图像
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.img_dim),
            nn.Sigmoid()  # 输出范围归一化到 [0, 1]
        )
        #########################################################
        #                     End of TODO                       #
        #########################################################
        # Move model to specified device
        self.to(device)

    def encode_param(self, x, y):
        """Compute mu and logstd of p(z|x, y)
        
        Args:
            x: Input image tensor [B, C*H*W]
            y: Input label tensor [B, label_size]
            
        Returns:
            mu: Mean of latent distribution
            logstd: Log standard deviation of latent distribution
        """
        # Ensure inputs are on the correct device
        x = x.to(self.device)
        y = y.to(self.device)
        
        x_enc = self.img_encoder(x)
        y_enc = self.label_encoder(y)
        
        h = torch.cat([x_enc, y_enc], dim=1)
        h = self.encoder(h)
        
        mu = self.fc_mean(h)
        logstd = self.fc_logstd(h)
        
        return mu, logstd

    def reparameterize(self, mu, logstd):
        """Reparameterization trick
        
        Args:
            mu: Mean of latent distribution
            logstd: Log standard deviation of latent distribution
            
        Returns:
            Sampled latent vector
        """
        
        #########################################################
        #                     TODO: Reparameterization         #
        #########################################################
        std = torch.exp(0.5 * logstd)

        # 从标准正态分布中采样 epsilon
        eps = torch.randn_like(std)

        # 使用 reparameterization trick 生成 z = mu + std * eps
        z = mu + std * eps
        #########################################################
        #                     End of TODO                       #
        #########################################################
        return z

    def encode(self, x, y):
        """Sample latent z from p(z|x, y)
        
        Args:
            x: Input image tensor [B, C*H*W]
            y: Input label tensor [B, label_size]
            
        Returns:
            z: Sampled latent vector
            mu: Mean of latent distribution
            logstd: Log standard deviation of latent distribution
        """
        # Ensure inputs are on the correct device
        x = x.to(self.device)
        y = y.to(self.device)
        
        mu, logstd = self.encode_param(x, y)
        z = self.reparameterize(mu, logstd)
        return z, mu, logstd

    def decode(self, z, y):
        """Decode latent vector and label to reconstruction
        
        Args:
            z: Latent vector
            y: Label tensor
            
        Returns:
            Reconstructed image
        """
        # Ensure inputs are on the correct device
        z = z.to(self.device)
        y = y.to(self.device)
        
        z_dec = self.z_decoder(z)
        y_dec = self.label_decoder(y)
        
        h = torch.cat([z_dec, y_dec], dim=1)
        reconstruction = self.decoder(h)
        
        return reconstruction

    def forward(self, x, y):
        """Forward pass through the VAE
        
        Args:
            x: Input image tensor [B, C*H*W]
            y: Input label tensor [B, label_size]
            
        Returns:
            reconstruction: Reconstructed image
            mu: Mean of latent distribution
            logstd: Log standard deviation of latent distribution
        """
        # Ensure inputs are on the correct device
        x = x.to(self.device)
        y = y.to(self.device)
        
        z, mu, logstd = self.encode(x, y)
        reconstruction = self.decode(z, y)
        return reconstruction, mu, logstd
    
    def train_step(self, batch, optimizer):
        """Perform a single training step
        
        Args:
            batch: Batch data from dataloader (x, y)
            optimizer: Optimizer for the model
            
        Returns:
            loss: Loss value for this batch
        """
        x, y = batch
        #########################################################
        #                  TODO: VAE Training Step              #
        #########################################################
        # Hints:
        # You can utilize the following pipeline:
        # 1. Flatten the image
        # 2. Convert label to one-hot
        # 3. Forward pass
        # 4. Calculate loss
        # 5. Backward pass & Optimizer step
        #########################################################
        batch_size = x.size(0)

        # Flatten the image
        x = x.view(batch_size, -1).to(self.device)

        # Convert label to one-hot
        y_onehot = torch.zeros(batch_size, self.label_size, device = self.device)
        y_onehot.scatter_(1, y.unsqueeze(1).to(self.device), 1)

        # Forward pass
        recon_x, mu, logstd = self(x, y_onehot)

        # Calculate loss
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction = 'sum')
        kl_loss = -0.5 * torch.sum(1 + logstd - mu.pow(2) - logstd.exp())
        loss = recon_loss + kl_loss

        # Backward pass & Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item() / batch_size
        #########################################################
        #                     End of TODO                       #
        #########################################################
    
    def inference(self, n_samples=64, labels=None):
        """Generate samples from random noise
        
        Args:
            n_samples: Number of samples to generate
            labels: Optional tensor of labels [n_samples, label_size]
                   If None, generates samples for all classes
            
        Returns:
            Generated samples
        """
        self.eval()
        
        # If no labels provided, generate samples for all classes
        if labels is None:
            # Create labels for each class
            n_per_class = n_samples // self.label_size
            labels = []
            for i in range(self.label_size):
                label = torch.zeros(n_per_class, self.label_size, device=self.device)
                label[:, i] = 1
                labels.append(label)
            labels = torch.cat(labels, dim=0)
        else:
            # Ensure labels are on the correct device
            labels = labels.to(self.device)
        
        with torch.no_grad():
            z = torch.randn(labels.shape[0], self.latent_size, device=self.device)
            samples = self.decode(z, labels)
            samples = samples.view(-1, *self.img_size)
        
        return samples
    
    def evaluate(self, dataloader):
        """Evaluate reconstruction quality
        
        Args:
            dataloader: DataLoader for evaluation
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        self.eval()
        total_recon_loss = 0
        total_kl_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                batch_size = x.size(0)
                x = x.view(batch_size, -1).to(self.device)
                
                # Convert label to one-hot
                y_onehot = torch.zeros(batch_size, self.label_size, device=self.device)
                y_onehot.scatter_(1, y.unsqueeze(1).to(self.device), 1)
                
                # Forward pass
                recon_x, mu, logstd = self(x, y_onehot)
                
                # Calculate losses
                recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logstd - mu.pow(2) - logstd.exp())
                
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                total_samples += batch_size
        
        avg_recon_loss = total_recon_loss / total_samples
        avg_kl_loss = total_kl_loss / total_samples
        avg_total_loss = avg_recon_loss + avg_kl_loss
        
        return {
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss,
            'total_loss': avg_total_loss
        }
    
    @torch.no_grad()
    def sample_images(self, label, save=True, save_dir='./vae'):
        """Sample images for given labels
        
        Args:
            label: Label tensor [B, label_size]
            save: Whether to save images
            save_dir: Directory to save images
            
        Returns:
            Generated images
        """
        self.eval()
        # Ensure label is on the correct device
        label = label.to(self.device)
        n_samples = label.shape[0]
        samples = self.decode(torch.randn(n_samples, self.latent_size, device=self.device), label)
        imgs = samples.view(n_samples, *self.img_size).clamp(0., 1.)
        if save:
            os.makedirs(save_dir, exist_ok=True)
            torchvision.utils.save_image(imgs, os.path.join(save_dir, 'sample.png'), nrow=int(n_samples**0.5))
        return imgs
    
    @torch.no_grad()
    def make_dataset(self, n_samples_per_class=10, save=True, save_dir='./vae/generated/'):
        """Generate dataset with samples for each class
        
        Args:
            n_samples_per_class: Number of samples per class
            save: Whether to save images
            save_dir: Directory to save images
            
        Returns:
            None
        """
        self.eval()
        for i in range(self.label_size):
            label = torch.zeros(n_samples_per_class, self.label_size, device=self.device)
            label[:, i] = 1
            samples = self.decode(torch.randn(n_samples_per_class, self.latent_size, device=self.device), label)
            imgs = samples.view(n_samples_per_class, *self.img_size).clamp(0., 1.)
            if save:
                os.makedirs(os.path.join(save_dir, str(i)), exist_ok=True)
                for j in range(n_samples_per_class):
                    torchvision.utils.save_image(imgs[j], os.path.join(save_dir, str(i), "{}_{:>03d}.png".format(i, j))) 
import os
import cv2
import torch
import einops
import numpy as np

def save_generated_images(samples, output_path, n_sample_per_side=10, range_type='diffusion'):
    """Save generated samples as a grid image
    
    Args:
        samples: Tensor of generated samples [B, C, H, W]
        output_path: Path to save the image
        n_sample_per_side: Number of samples per side of the grid
        range_type: Type of range for the samples ('diffusion' [-1, 1] or 'vae' [0, 1])
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Process samples based on range type
    if range_type == 'diffusion':
        # Convert from [-1, 1] to [0, 255]
        samples = ((samples + 1) / 2) * 255
    else:  # vae
        # Convert from [0, 1] to [0, 255]
        samples = samples * 255
    
    samples = samples.clamp(0, 255)
    
    # Arrange samples in a grid
    samples = einops.rearrange(
        samples, '(b1 b2) c h w -> (b1 h) (b2 w) c', 
        b1=n_sample_per_side
    )
    
    # Convert to numpy array
    image = samples.cpu().numpy().astype(np.uint8)
    
    # Save the image
    cv2.imwrite(output_path, image)
    
    return image 
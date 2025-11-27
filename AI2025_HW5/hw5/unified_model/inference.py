import os
import argparse
import torch

from unified_model.data.mnist_data import get_mnist_tensor_shape
from unified_model.models.unet import UNet, ConditionalUNet
from unified_model.models.diffusion_model import DiffusionModel, ConditionalDiffusionModel
from unified_model.models.vae_model import CVAE
from unified_model.trainers.diffusion_trainer import DiffusionTrainer
from unified_model.trainers.vae_trainer import VAETrainer
from unified_model.utils.image_utils import save_generated_images

def parse_args():
    """Parse command line arguments for inference"""
    parser = argparse.ArgumentParser(description='Inference script for generative models')
    
    # General arguments
    parser.add_argument('--model', type=str, default='diffusion', 
                        choices=['diffusion', 'vae', 'conditional_diffusion'],
                        help='Model type to use')
    parser.add_argument('--device', type=str, default=None, 
                        help='Device to use (default: auto-detect)')
    parser.add_argument('--output_dir', type=str, default='out', help='Output directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to load model from')
    
    # Inference arguments
    parser.add_argument('--n_samples', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--save_by_class', action='store_true', 
                        help='Save samples by class (for organization)')
    parser.add_argument('--n_samples_per_class', type=int, default=10,
                        help='Number of samples per class when using save_by_class')
    parser.add_argument('--specific_class', type=int, default=None,
                        help='Generate samples for a specific class (only for conditional models)')
    
    # Diffusion model arguments
    parser.add_argument('--sampler', type=str, default='ddpm', choices=['ddpm'],
                        help='Sampler type for diffusion model')
    parser.add_argument('--n_steps', type=int, default=1000, help='Number of diffusion steps')
    
    # Conditional diffusion arguments
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes (for conditional models)')
    parser.add_argument('--label_emb_dim', type=int, default=32, help='Size of label embedding (for conditional models)')
    
    # VAE model arguments
    parser.add_argument('--latent_size', type=int, default=128, help='Size of latent space for VAE')
    parser.add_argument('--hidden_size', type=int, default=512, help='Size of hidden layers for VAE')
    
    return parser.parse_args()

def main():
    """Main inference function"""
    args = parse_args()
    
    # Auto-detect device if not specified
    if args.device is None:
        args.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model based on type
    if args.model == 'diffusion':
        # Initialize diffusion model
        print("Initializing Diffusion model...")
        unet = UNet(args.n_steps, device=args.device)
        model = DiffusionModel(
            unet, 
            sampler_type=args.sampler, 
            n_steps=args.n_steps, 
            device=args.device
        )
        trainer = DiffusionTrainer(
            model, 
            args.device, 
            os.path.join(args.output_dir, args.model)
        )
    
    elif args.model == 'conditional_diffusion':
        # Initialize conditional diffusion model
        print("Initializing Conditional Diffusion model...")
        conditional_unet = ConditionalUNet(
            args.n_steps, 
            num_classes=args.num_classes,
            label_emb_dim=args.label_emb_dim,
            device=args.device
        )
        model = ConditionalDiffusionModel(
            conditional_unet,
            num_classes=args.num_classes,
            sampler_type=args.sampler, 
            n_steps=args.n_steps, 
            device=args.device
        )
        trainer = DiffusionTrainer(
            model, 
            args.device, 
            os.path.join(args.output_dir, args.model)
        )
        
    elif args.model == 'vae':
        # Initialize VAE model
        print("Initializing VAE model...")
        img_size = get_mnist_tensor_shape()
        label_size = 10  # MNIST has 10 classes
        model = CVAE(
            img_size=img_size,
            label_size=label_size,
            latent_size=args.latent_size,
            hidden_size=args.hidden_size,
            device=args.device
        )
        trainer = VAETrainer(
            model, 
            args.device, 
            os.path.join(args.output_dir, args.model)
        )
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    trainer.load_model(args.model_path)
    
    # Inference
    print(f"Generating samples using {args.model} model...")
    
    # Handle specific class request for conditional models
    if args.specific_class is not None and args.model == 'conditional_diffusion':
        print(f"Generating {args.n_samples} samples for class {args.specific_class}...")
        save_dir = os.path.join(args.output_dir, f"{args.model}_class_{args.specific_class}")
        samples = model.sample_class_images(
            args.specific_class, 
            n_samples=args.n_samples, 
            save=True, 
            save_dir=save_dir
        )
        print(f"Samples saved to {save_dir}")
        return
    
    if args.save_by_class:
        # Generate and save samples by class
        print(f"Generating {args.n_samples_per_class} samples per class...")
        save_dir = os.path.join(args.output_dir, f"{args.model}_samples_by_class")
        model.make_dataset(n_samples_per_class=args.n_samples_per_class, save=True, save_dir=save_dir)
        print(f"Samples saved to {save_dir}")
    else:
        # Generate samples as a grid
        if args.model == 'diffusion':
            n_sample_per_side = int(args.n_samples ** 0.5)
            shape = (n_sample_per_side**2, *get_mnist_tensor_shape())
            samples = model.inference(shape)
            
            # Save as a grid
            output_path = f"{args.output_dir}/{args.model}_{args.sampler}_samples.png"
            save_generated_images(samples, output_path, n_sample_per_side, range_type='diffusion')
            print(f"Samples saved to {output_path}")
        
        elif args.model == 'conditional_diffusion':
            # Sample images with random labels
            n_sample_per_side = int(args.n_samples ** 0.5)
            samples = model.sample_images(
                n_samples=n_sample_per_side**2, 
                save=True, 
                save_dir=os.path.join(args.output_dir, args.model)
            )
            print(f"Samples saved to {os.path.join(args.output_dir, args.model)}")
            
        else:  # vae
            # Create one-hot labels for each class
            n_per_class = args.n_samples // 10
            labels = []
            for i in range(10):  # MNIST has 10 classes
                label = torch.zeros(n_per_class, 10, device=args.device)
                label[:, i] = 1
                labels.append(label)
            labels = torch.cat(labels, dim=0)
            
            # Generate samples
            samples = model.inference(args.n_samples, labels)
            
            # Save as a grid
            output_path = f"{args.output_dir}/{args.model}_samples.png"
            save_generated_images(samples, output_path, int(args.n_samples**0.5), range_type='vae')
            print(f"Samples saved to {output_path}")

if __name__ == '__main__':
    main() 
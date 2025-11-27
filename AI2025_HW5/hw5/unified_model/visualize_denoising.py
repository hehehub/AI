import os
import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from unified_model.data.mnist_data import get_mnist_tensor_shape
from unified_model.models.unet import UNet, ConditionalUNet
from unified_model.models.diffusion_model import DiffusionModel, ConditionalDiffusionModel
from unified_model.trainers.diffusion_trainer import DiffusionTrainer
from unified_model.samplers.diffusion_samplers import DDPM
from unified_model.utils.image_utils import save_generated_images

def parse_args():
    """Parse command line arguments for denoising visualization"""
    parser = argparse.ArgumentParser(description='Visualize denoising process of diffusion models')
    
    # General arguments
    parser.add_argument('--model', type=str, default='diffusion', 
                        choices=['diffusion', 'conditional_diffusion'],
                        help='Model type to use')
    parser.add_argument('--device', type=str, default=None, 
                        help='Device to use (default: auto-detect)')
    parser.add_argument('--output_dir', type=str, default='out/visualization', 
                        help='Output directory')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to load model from')
    
    # Visualization arguments
    parser.add_argument('--n_samples', type=int, default=5, 
                        help='Number of samples to visualize')
    parser.add_argument('--n_steps_to_show', type=int, default=10, 
                        help='Number of denoising steps to show')
    parser.add_argument('--specific_class', type=int, default=None,
                        help='Generate samples for a specific class (only for conditional models)')
    parser.add_argument('--save_video', action='store_true',
                        help='Save the denoising process as a video')
    parser.add_argument('--fps', type=int, default=5,
                        help='Frames per second for the video')
    
    # Diffusion model arguments
    parser.add_argument('--sampler', type=str, default='ddpm', choices=['ddpm'],
                        help='Sampler type for diffusion model')
    parser.add_argument('--n_steps', type=int, default=1000, 
                        help='Number of diffusion steps')
    
    # Conditional diffusion arguments
    parser.add_argument('--num_classes', type=int, default=10, 
                        help='Number of classes (for conditional models)')
    parser.add_argument('--label_emb_dim', type=int, default=32, 
                        help='Size of label embedding (for conditional models)')
    
    return parser.parse_args()

def visualize_denoising_process(model, sampler, n_samples, n_steps_to_show, device, 
                               save_dir, specific_class=None, save_video=False, fps=5):
    """Visualize the denoising process of a diffusion model
    
    Args:
        model: Diffusion model
        sampler: Diffusion sampler (DDPM)
        n_samples: Number of samples to visualize
        n_steps_to_show: Number of denoising steps to show
        device: Device to use
        save_dir: Directory to save visualizations
        specific_class: Specific class to generate (for conditional models)
        save_video: Whether to save the denoising process as a video
        fps: Frames per second for the video
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # Initialize noise
    shape = (n_samples, *get_mnist_tensor_shape())
    x = torch.randn(shape).to(device)
    
    # For conditional models, prepare labels
    labels = None
    if hasattr(model, 'num_classes'):  # Check if it's a conditional model
        if specific_class is not None:
            # Use the specific class for all samples
            labels = torch.tensor([specific_class] * n_samples, dtype=torch.long).to(device)
        else:
            # Use different classes for each sample (cycling through 0-9)
            labels = torch.tensor([i % model.num_classes for i in range(n_samples)], dtype=torch.long).to(device)
    
    # Get the total number of steps from the sampler
    total_steps = sampler.n_steps
    
    # Calculate which steps to visualize
    if isinstance(sampler, DDPM):
        # For DDPM, create steps with exact intervals of 100
        # 生成[1000, 900, 800, ..., 100, 0]
        steps_to_visualize = []
        for step in range(total_steps, -1, -100):
            steps_to_visualize.append(step)
        # 确保包含最后一步（0）
        if steps_to_visualize[-1] != 0:
            steps_to_visualize.append(0)
        # 调整步数以匹配n_steps_to_show
        if len(steps_to_visualize) > n_steps_to_show:
            # 如果步骤太多，均匀选择
            indices = np.round(np.linspace(0, len(steps_to_visualize) - 1, n_steps_to_show)).astype(int)
            steps_to_visualize = [steps_to_visualize[i] for i in indices]
        step_indices = np.array(steps_to_visualize)
    else:
        raise ValueError(f"Unsupported sampler type: {type(sampler)}")
    
    # Initialize list to store frames for video
    frames = []
    
    # Create a figure for visualization
    plt.figure(figsize=(n_steps_to_show * 2, n_samples * 2))
    
    # Perform denoising and visualize intermediate steps
    x_steps = []
    
    # Custom denoising function for visualization
    if isinstance(sampler, DDPM):
        # For DDPM
        net = model.unet
        net = net.to(device)
        
        # Store initial noisy images
        x_steps.append(x.clone())
        step_names = ["initial_noise"]
        
        # Denoise step by step
        for t in tqdm(range(total_steps - 1, -1, -1), desc="Denoising"):
            # Denoising step
            with torch.no_grad():
                if labels is not None:
                    # For conditional model
                    t_tensor = torch.tensor([t] * x.shape[0], dtype=torch.long).to(device).unsqueeze(1)
                    # 创建一个自定义的去噪函数，正确处理条件标签
                    def denoise_fn(x_in, t_in):
                        # 检查t_in的维度
                        if len(t_in.shape) == 1:
                            # 如果t是一维的，为UNet准备正确的格式
                            return net(x_in, t_in.unsqueeze(1), labels)
                        else:
                            # 如果t已经有正确的维度
                            return net(x_in, t_in, labels)
                    
                    eps_t = denoise_fn(x, t_tensor)
                    x = sampler.sample_backward_t(denoise_fn, x, t)
                else:
                    # For unconditional model
                    t_tensor = torch.tensor([t] * x.shape[0], dtype=torch.long).to(device).unsqueeze(1)
                    eps_t = net(x, t_tensor)
                    x = sampler.sample_backward_t(net, x, t)
            
            # Save intermediate steps
            if t in step_indices:
                x_steps.append(x.clone())
                step_names.append(f"step_{t}")
        
        # 确保最后一步被保存
        if 0 not in step_indices and t == 0:
            x_steps.append(x.clone())
            step_names.append("final_result")
    
    else:
        raise ValueError(f"Unsupported sampler type: {type(sampler)}")
    
    # Convert tensors to images and visualize
    for i, step_images in enumerate(x_steps):
        # Convert from [-1, 1] to [0, 1]
        step_images = ((step_images + 1) / 2).clamp(0, 1)
        # 为视频添加帧
        img = (step_images.cpu().numpy() * 255).astype(np.uint8)
        img = np.transpose(img, (0, 2, 3, 1))  # [B, H, W, C]
        frames.append(img)
    
    # Create a grid visualization of all steps
    fig, axes = plt.subplots(n_samples, len(x_steps), figsize=(len(x_steps) * 3, n_samples * 3))
    
    for i in range(n_samples):
        for j in range(len(x_steps)):
            ax = axes[i, j] if n_samples > 1 else axes[j]
            img = frames[j][i]
            
            # Convert to RGB if grayscale
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            
            if i == 0:
                ax.set_title(step_names[j])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "denoising_grid.png"), dpi=300)
    plt.close()
    
    # Save as video if requested
    if save_video:
        create_denoising_video(frames, os.path.join(save_dir, "denoising_process.mp4"), fps)
    
    print(f"Visualization saved to {save_dir}")

def create_denoising_video(frames, output_path, fps=5):
    """Create a video of the denoising process
    
    Args:
        frames: List of frames [steps, batch, H, W, C]
        output_path: Path to save the video
        fps: Frames per second
    """
    n_samples = frames[0].shape[0]
    h, w = frames[0].shape[1:3]
    
    # 增加分辨率 - 将图像放大到更高的分辨率
    scale_factor = 4  # 放大4倍
    new_h, new_w = h * scale_factor, w * scale_factor
    
    # Create a video for each sample
    for i in range(n_samples):
        sample_frames = [frame[i] for frame in frames]
        
        # Create video writer - 使用更高的分辨率
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        video_path = output_path.replace('.mp4', f'_sample_{i}.mp4')
        out = cv2.VideoWriter(video_path, fourcc, fps, (new_w, new_h))
        
        # Write frames
        for frame in sample_frames:
            # Convert to BGR for OpenCV
            if frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
            
            # 放大图像到更高的分辨率
            frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # 确保像素值在0-255范围内
            frame_resized = np.clip(frame_resized, 0, 255).astype(np.uint8)
            
            # 转为BGR并写入
            frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
    
    print(f"Videos saved to {os.path.dirname(output_path)} with resolution {new_w}x{new_h}")

def main():
    """Main function for denoising visualization"""
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
            None  # 设置为None，不创建额外的输出目录
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
            None  # 设置为None，不创建额外的输出目录
        )
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    trainer.load_model(args.model_path)
    
    # Visualize denoising process
    print(f"Visualizing denoising process using {args.model} model with {args.sampler} sampler...")
    visualize_denoising_process(
        model,
        model.sampler,
        args.n_samples,
        args.n_steps_to_show,
        args.device,
        os.path.join(args.output_dir, f"{args.model}_{args.sampler}_visualization"),
        args.specific_class,
        args.save_video,
        args.fps
    )

if __name__ == '__main__':
    main() 
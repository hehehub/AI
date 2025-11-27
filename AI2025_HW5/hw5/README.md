# 人工智能导论第5次作业·编程部分

本项目提供了一个统一的框架，用于训练、推理和评估不同类型的生成模型，包括条件变分自编码器（CVAE）、扩散模型（Diffusion Models）和条件扩散模型（Conditional Diffusion Models）。

在本项目中，你需要按照作业要求，补全对应的代码空缺，并运行对应的训练/推理/可视化/评测脚本。需要补全的内容会以如下格式的注释框包裹:
```py
#########################################################
#                  TODO: Example                        #
#########################################################
# Hints:
# Placeholder
#########################################################
<Some code you need to fill in>
#########################################################
#                     End of TODO                       #
#########################################################
```
## 依赖

- PyTorch
- torchvision
- numpy
- opencv-python
- einops
- tqdm
- scipy
- matplotlib 

## 项目结构

```
unified_model/
├── data/               # 数据加载和处理
├── models/             # 模型定义
├── trainers/           # 训练器
├── samplers/           # 采样器（用于扩散模型）
├── utils/              # 工具函数
├── train.py            # 训练脚本
├── inference.py        # 推理脚本
└── visualize_denoising.py  # 扩散模型去噪过程可视化脚本
cal_fid.py              # 计算FID指标
```

## 支持的模型

- **条件变分自编码器 (CVAE)**: 基于MNIST数据集的条件VAE实现
- **扩散模型 (Diffusion)**: 包括DDPM采样器的实现
- **条件扩散模型 (Conditional Diffusion)**: 基于类别标签的条件扩散模型实现

## 使用方法

### 训练模型

#### 训练CVAE模型

```bash
python -m unified_model.train --model vae --batch_size 128 --n_epochs 50 --output_dir out/vae --latent_size 100 --hidden_size 256 --beta 1.0
```

#### 训练Diffusion模型

```bash
python -m unified_model.train --model diffusion --batch_size 512 --n_epochs 100 --output_dir out/diffusion --sampler ddpm --n_steps 1000
```

#### 训练条件Diffusion模型

```bash
python -m unified_model.train --model conditional_diffusion --batch_size 512 --n_epochs 100 --output_dir out/cond_diffusion --sampler ddpm --n_steps 1000 --num_classes 10 --label_emb_dim 32
```

### 生成样本

#### 使用CVAE生成样本

```bash
# 生成网格图像（out/samples/vae_samples.png）
python -m unified_model.inference --model vae --model_path out/vae/vae/vae_best.pth --n_samples 100 --output_dir out/samples

# 按类别生成图像（用于计算FID等指标）
python -m unified_model.inference --model vae --model_path out/vae/vae/vae_best.pth --save_by_class --n_samples_per_class 100 --output_dir out/samples
```

#### 使用Diffusion模型生成样本

```bash
# 生成网格图像
python -m unified_model.inference --model diffusion --model_path out/diffusion/diffusion/epoch_100/model.pth --sampler ddpm --n_steps 1000 --n_samples 100 --output_dir out/samples
```

#### 使用条件Diffusion模型生成样本

```bash
# 生成网格图像(每一排一种标签)
python -m unified_model.inference --model conditional_diffusion --model_path out/cond_diffusion/conditional_diffusion/epoch_100/model.pth --sampler ddpm --n_steps 1000 --n_samples 100 --output_dir out/samples

# 生成特定类别的图像
python -m unified_model.inference --model conditional_diffusion --model_path out/cond_diffusion/conditional_diffusion/epoch_100/model.pth --sampler ddpm --n_steps 1000 --specific_class 0 --n_samples 25 --output_dir out/samples

# 按类别生成图像（用于计算FID等指标）
python -m unified_model.inference --model conditional_diffusion --model_path out/cond_diffusion/conditional_diffusion/epoch_100/model.pth --sampler ddpm --n_steps 1000 --save_by_class --n_samples_per_class 100 --output_dir out/samples
```

### 可视化扩散模型的去噪过程

#### 使用DDPM采样器可视化去噪过程

```bash
# 可视化标准扩散模型的去噪过程
python -m unified_model.visualize_denoising --model diffusion --model_path out/diffusion/diffusion/epoch_100/model.pth --sampler ddpm --n_steps 1000 --n_samples 5 --n_steps_to_show 10 --output_dir out/visualization

# 保存去噪过程的视频
python -m unified_model.visualize_denoising --model diffusion --model_path out/diffusion/diffusion/epoch_100/model.pth --sampler ddpm --n_steps 1000 --n_samples 5 --n_steps_to_show 10 --save_video --output_dir out/visualization
```

#### 可视化条件扩散模型的去噪过程

```bash
# 可视化条件扩散模型的去噪过程（随机类别）
python -m unified_model.visualize_denoising --model conditional_diffusion --model_path out/cond_diffusion/conditional_diffusion/epoch_100/model.pth --sampler ddpm --n_steps 1000 --n_samples 5 --n_steps_to_show 10 --output_dir out/visualization

# 可视化特定类别的去噪过程
python -m unified_model.visualize_denoising --model conditional_diffusion --model_path out/cond_diffusion/conditional_diffusion/epoch_100/model.pth --sampler ddpm --n_steps 1000 --n_samples 5 --n_steps_to_show 10 --specific_class 0 --output_dir out/visualization
```

### 计算FID评估指标
#### VAE
```
python cal_fid.py --vae
```
#### 条件扩散模型
```
python cal_fid.py --cond_diffusion
```
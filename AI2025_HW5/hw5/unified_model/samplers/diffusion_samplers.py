import torch

def sqrt(x):
    if isinstance(x, torch.Tensor):
        return torch.sqrt(x)
    return x ** 0.5

class DDPM:
    """Denoising Diffusion Probabilistic Model sampler"""

    def __init__(self, n_steps: int, device: str, min_beta: float = 0.0001, max_beta: float = 0.02):
        """
        Args:
            n_steps: Number of diffusion steps
            device: Device to use
            min_beta: Minimum noise level
            max_beta: Maximum noise level
        """
        self.n_steps = n_steps
        self.device = device
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.empty_like(self.alphas)
        product = 1
        for i, alpha in enumerate(self.alphas):
            product *= alpha
            self.alpha_bars[i] = product

    def sample_forward(self, x_0, t, noise=None):
        """Forward diffusion process (adding noise)
        
        Args:
            x_0: Original clean data
            t: Timestep
            noise: Optional noise to add (if None, random noise is used)
            
        Returns:
            Noisy data at timestep t
        """
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        #########################################################
        #                     TODO: Forward Process         #
        #########################################################
        # Hint: 
        # What do we want here? 
        # We want to add noise to the original data.
        # Think about the case when noise is not None.
        #########################################################
        if noise is None:
            eps = torch.randn_like(x_0)
        else:
            eps = noise.to(self.device)
        #########################################################
        #                     End of TODO                       #
        #########################################################
        res = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * eps  # x_t = sqrt(alpha_bar) * x_0  + sqrt(1 - alpha_bar) * eps
        return res

    def sample_backward(self, net, in_shape):
        """Backward diffusion process (removing noise)
        
        Args:
            net: Neural network model or function for noise prediction
            in_shape: Shape of the input tensor
            
        Returns:
            Generated samples
        """
        x = torch.randn(in_shape).to(self.device)
        
        # Handle both model objects and function-style denoisers
        if callable(net) and not hasattr(net, 'to'):
            # It's a function-style denoiser
            denoise_fn = net
        else:
            # It's a model object
            net = net.to(self.device)
            denoise_fn = lambda x_t, t: net(x_t, t)
        
        for t in range(self.n_steps - 1, -1, -1):
            x = self.sample_backward_t(denoise_fn, x, t)  # x_{t-1} = sample_backward_t(x_t)
        return x  # x_0

    def sample_backward_t(self, net, x_t, t):
        """Single step of backward diffusion
        
        Args:
            net: Neural network model or function for noise prediction
            x_t: Noisy data at timestep t
            t: Timestep
            
        Returns:
            Denoised data at timestep t-1
        """
        # Handle both model objects and function-style denoisers
        if callable(net) and not hasattr(net, 'to'):
            # It's a function-style denoiser
            t_tensor = torch.tensor([t] * x_t.shape[0], dtype=torch.long).to(x_t.device).unsqueeze(1)
            eps_t = net(x_t, t_tensor)
        else:
            # It's a model object
            t_tensor = torch.tensor([t] * x_t.shape[0], dtype=torch.long).to(x_t.device).unsqueeze(1)
            eps_t = net(x_t, t_tensor)
            
        mu_t = (x_t - (1 - self.alphas[t]) / sqrt(1 - self.alpha_bars[t]) * eps_t) / sqrt(self.alphas[t])  # posterior mean
        if t == 0:
            noise_t = 0
        else:
            beta_t = self.betas[t]
            beta_tilde_t =  (1 - self.alpha_bars[t - 1]) / (1 - self.alpha_bars[t]) * beta_t  # posterior variance
            noise_t = sqrt(beta_tilde_t) * torch.randn_like(x_t)
        x_t_minus_1 = mu_t + noise_t  # x_{t-1} = N(x_t-1; mu(x_t, x_0), beta(_tilde)_t I)
        return x_t_minus_1
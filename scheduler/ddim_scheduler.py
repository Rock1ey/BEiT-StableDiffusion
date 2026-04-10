import torch
import numpy as np


class DDIMScheduler:
    r"""
    DDIM (Denoising Diffusion Implicit Models) scheduler.
    Compatible with DDPM-trained models — no retraining needed.
    Allows trading off quality for speed by reducing inference steps.
    """

    def __init__(self, num_train_timesteps, beta_start, beta_end, num_inference_steps=50, eta=0.0):
        """
        :param num_train_timesteps: Total training timesteps (e.g. 1000)
        :param beta_start: Beta schedule start
        :param beta_end: Beta schedule end
        :param num_inference_steps: Number of denoising steps at inference (e.g. 50)
        :param eta: Stochasticity parameter. 0 = deterministic DDIM, 1 = DDPM equivalent
        """
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.eta = eta

        # Same beta schedule as LinearNoiseScheduler (compvis style)
        self.betas = (
            torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps) ** 2
        )
        self.alphas = 1.0 - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)

        # Build sub-sampled timestep sequence (evenly spaced)
        self.timesteps = self._build_timesteps(num_inference_steps)

    def _build_timesteps(self, num_inference_steps):
        """Create evenly spaced timestep subsequence from [0, T-1]."""
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round().astype(np.int64)
        timesteps = np.flip(timesteps).copy()  # Reverse: from T to 0
        return torch.from_numpy(timesteps).long()

    def sample_prev_timestep(self, xt, noise_pred, t_idx):
        """
        DDIM reverse step.
        :param xt: Current noisy sample [B, C, H, W]
        :param noise_pred: Predicted noise from model [B, C, H, W]
        :param t_idx: Index into self.timesteps (NOT the raw timestep value)
        :return: (x_{t-1}, x0_pred)
        """
        t = self.timesteps[t_idx]
        device = xt.device

        alpha_t = self.alpha_cum_prod[t].to(device)
        alpha_t_prev = self.alpha_cum_prod[self.timesteps[t_idx + 1]].to(device) \
            if t_idx + 1 < len(self.timesteps) else torch.tensor(1.0, device=device)

        # Predict x0
        x0_pred = (xt - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        # Compute sigma for stochasticity control
        sigma = self.eta * torch.sqrt(
            (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
        )

        # Direction pointing to x_t
        dir_xt = torch.sqrt(torch.clamp(1 - alpha_t_prev - sigma ** 2, min=0)) * noise_pred

        # DDIM update rule
        x_prev = torch.sqrt(alpha_t_prev) * x0_pred + dir_xt

        if self.eta > 0 and t_idx + 1 < len(self.timesteps):
            noise = torch.randn_like(xt)
            x_prev = x_prev + sigma * noise

        return x_prev, x0_pred

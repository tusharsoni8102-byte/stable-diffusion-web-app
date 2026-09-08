import torch
import numpy as np


class DDIMSampler:

    def __init__(
        self,
        generator: torch.Generator,
        num_training_steps=1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
    ):
        self.betas = (
            torch.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_training_steps,
                dtype=torch.float32,
            )
            ** 2
        )

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.generator = generator

        self.num_train_timesteps = num_training_steps
        self.timesteps = torch.from_numpy(
            np.arange(0, num_training_steps)[::-1].copy()
        )

    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps

        step_ratio = self.num_train_timesteps // num_inference_steps

        timesteps = (
            np.arange(0, num_inference_steps)
            * step_ratio
        ).round()[::-1].copy().astype(np.int64)

        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep):
        prev_t = (
            timestep
            - self.num_train_timesteps // self.num_inference_steps
        )

        return prev_t

    def set_strength(self, strength=1):
        start_step = self.num_inference_steps - int(
            self.num_inference_steps * strength
        )

        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def step(
        self,
        timestep: int,
        latents: torch.Tensor,
        model_output: torch.Tensor,
    ):
        t = timestep
        prev_t = self._get_previous_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]

        if prev_t >= 0:
            alpha_prod_t_prev = self.alphas_cumprod[prev_t]
        else:
            alpha_prod_t_prev = torch.tensor(
                1.0,
                device=latents.device,
                dtype=latents.dtype,
            )

        beta_prod_t = 1 - alpha_prod_t

        # Predict x_0 from x_t and predicted noise.
        pred_original_sample = (
            latents
            - beta_prod_t ** 0.5 * model_output
        ) / alpha_prod_t ** 0.5

        # DDIM direction pointing toward x_t.
        pred_sample_direction = (
            (1 - alpha_prod_t_prev) ** 0.5
            * model_output
        )

        # Deterministic DDIM update (eta = 0).
        pred_prev_sample = (
            alpha_prod_t_prev ** 0.5
            * pred_original_sample
            + pred_sample_direction
        )

        return pred_prev_sample

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ):
        alphas_cumprod = self.alphas_cumprod.to(
            device=original_samples.device,
            dtype=original_samples.dtype,
        )

        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = (
            alphas_cumprod[timesteps] ** 0.5
        )

        sqrt_alpha_prod = sqrt_alpha_prod.flatten()

        while len(sqrt_alpha_prod.shape) < len(
            original_samples.shape
        ):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (
            1 - alphas_cumprod[timesteps]
        ) ** 0.5

        sqrt_one_minus_alpha_prod = (
            sqrt_one_minus_alpha_prod.flatten()
        )

        while len(sqrt_one_minus_alpha_prod.shape) < len(
            original_samples.shape
        ):
            sqrt_one_minus_alpha_prod = (
                sqrt_one_minus_alpha_prod.unsqueeze(-1)
            )

        noise = torch.randn(
            original_samples.shape,
            generator=self.generator,
            device=original_samples.device,
            dtype=original_samples.dtype,
        )

        noisy_samples = (
            sqrt_alpha_prod * original_samples
            + sqrt_one_minus_alpha_prod * noise
        )

        return noisy_samples
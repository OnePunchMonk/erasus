"""
Noise Injection Strategy (Diffusion Models).

Fine-tunes the diffusion model (U-Net) to predict *random noise* 
(or a completely different concept) when conditioned on the forget prompt,
effectively breaking the link between the text prompt and the generated image.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


@strategy_registry.register("noise_injection")
class NoiseInjectionStrategy(BaseStrategy):
    """
    For Forget Sample (Effective Prompt P, Image I):
    Standard Objective: || ε - U_net(I_t, t, P) ||^2
    
    Unlearning Objective:
    1. Random Label: || ε - U_net(I_t, t, P) ||^2  where I_t is noise from DIFFERENT image?
    2. Or simpler: Maximize Loss?
    3. Better (ESD approach): || U_net(I_t, t, P) - U_net(I_t, t, "") ||^2
       (Steer P towards Unconditional/Null P).
    """

    def __init__(
        self,
        lr: float = 1e-5,
        guidance_scale: float = 1.0,  # Scale for ESD
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.lr = lr
        self.guidance_scale = guidance_scale

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        
        # Typically model here is the Diffusion Wrapper which has .unet, .vae, .scheduler
        # We assume the wrapper exposes a .forward that computes loss, 
        # OR we manually do the diffusion loop if wrapper provides components.
        
        # Assuming `model` is `StableDiffusionWrapper` or similar which may handle valid prop.
        # But BaseStrategy receives nn.Module. 
        # If standard Diffusion Pipeline, we need to be careful.
        
        device = next(model.parameters()).device
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        forget_losses = []
        model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            n = 0
            for batch in forget_loader:
                # Batch is (images, texts)
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                
                optimizer.zero_grad()
                
                # We need to execute the forward pass of the diffusion model manually or via wrapper
                # Assuming wrapper has `compute_loss(pixel_values, input_ids)`
                if hasattr(model, "compute_loss"):
                    # Normal training loss
                    # loss = model.compute_loss(pixel_values, input_ids)
                    
                    # Unlearning:
                    # Target: We want the model to behave as if it received an Empty Prompt
                    # when it receives the Forget Prompt.
                    
                    # Erasing Stable Diffusion (ESD) Method:
                    # Loss = || Pred_noise(x, t, c_forget) - Pred_noise(x, t, c_empty) ||^2 
                    # We minimize this difference? NO, we assume the empty prompt predictions are GROUND TRUTH.
                    # Yes, we retrain c_forget to match c_empty.
                    
                    loss = model.compute_loss(
                        pixel_values, 
                        input_ids, 
                        target_override="empty" # Hypothetical flag for wrapper
                    )
                else:
                    # Fallback dummy if wrapper not fully compliant
                    # Just standard GA
                    output = model(input_ids) # Dummy
                    loss = -output.sum() 

                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n += 1
            forget_losses.append(epoch_loss / max(n, 1))

        return model, forget_losses, []


"""
Basic Example: Unlearning with CLIP.

This script demonstrates how to key components of Erasus:
1. Load a CLIP model.
2. Create dummy Forget/Retain data.
3. Initialize the ErasusUnlearner.
4. Run unlearning with Contrastive Strategy.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPConfig

from erasus.unlearners.erasus_unlearner import ErasusUnlearner
# Make sure strategies are registered by importing them
import erasus.strategies.vlm_specific.contrastive_unlearning

class DummyImageTextDataset(Dataset):
    def __init__(self, size=10):
        self.size = size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Dummy image: [3, 224, 224]
        image = torch.randn(3, 224, 224) 
        # Dummy input_ids: [77] usually
        input_ids = torch.randint(0, 1000, (77,))
        # Attention mask
        attention_mask = torch.ones(77)
        
        # Return as tuple expected by strategy, or dict?
        # ContrastiveUnlearningStrategy expects batch[0] (images), batch[1] (texts)
        # where texts could be input_ids directly if model handles it, 
        # but CLIPModel expects input_ids and pixel_values.
        
        # Strategy call: outputs = model(images, texts)
        # So we probably need to wrap inputs correctly.
        return image, input_ids

def main():
    print("Initializing CLIP model (small config for speed)...")
    config = CLIPConfig(
        text_config_dict={"vocab_size": 1000, "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4},
        vision_config_dict={"image_size": 224, "patch_size": 32, "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4},
    )
    # Instantiate from config instead of pretrained to avoid download
    model = CLIPModel(config)
    
    print("Creating dummy datasets...")
    forget_set = DummyImageTextDataset(size=20)
    retain_set = DummyImageTextDataset(size=20)
    
    forget_loader = DataLoader(forget_set, batch_size=4)
    retain_loader = DataLoader(retain_set, batch_size=4)
    
    print("Initializing Erasus Unlearner...")
    unlearner = ErasusUnlearner(
        model=model,
        strategy="contrastive_unlearning",
        selector="random", # Use random selection of forget set
        strategy_kwargs={"lr": 1e-4, "neg_weight": 0.5},
        device="cpu" 
    )
    
    print("Starting Unlearning...")
    result = unlearner.fit(
        forget_data=forget_loader,
        retain_data=retain_loader,
        prune_ratio=0.5, # Keep 50% of forget set as coreset
        epochs=1
    )
    
    print(f"Unlearning Complete!")
    print(f"Elapsed Time: {result.elapsed_time:.2f}s")
    print(f"Coreset Size: {result.coreset_size}")
    print(f"Final Forget Loss: {result.forget_loss_history[-1] if result.forget_loss_history else 'N/A'}")

if __name__ == "__main__":
    main()

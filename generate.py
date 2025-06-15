import torch
import torchvision.utils as vutils
import os
import time


def generate_images(model_dir: str, out:str, qty:int, device:str = 'cuda'):
    device = torch.device(device)

    G = torch.jit.load(model_dir).to(device)
    G.eval()

    z = torch.randn(qty, 100, device=device)
    with torch.no_grad():
        fakes = G(z)

    os.makedirs(out, exist_ok=True)
    
    for i, img in enumerate(fakes):
        vutils.save_image(
            img,
            os.path.join(out, f"Generated_{i}_{int(time.time())}.png"),
            normalize=True,   
            value_range=(-1, 1),          
        )
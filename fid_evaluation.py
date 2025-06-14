import torch
import os
import torchvision.utils as vutils
from torchvision import transforms
from tqdm import tqdm
import json
from pytorch_fid import fid_score as fid_score_func 

def generate_fake_images(generator, latent_dim, save_dir, n_samples=50000, batch_size=100, device='cuda'):
    os.makedirs(save_dir, exist_ok=True)
    generator.eval().to(device)
    
    n_batches = n_samples // batch_size
    img_id = 0

    for _ in tqdm(range(n_batches)):
        z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        with torch.no_grad():
            fakes = generator(z).detach().cpu()  
            fakes = fakes.add(1).div(2).clamp(0, 1)  
        
        for img in fakes:
            vutils.save_image(img, os.path.join(save_dir, f"{img_id:05}.png"))
            img_id += 1

def dump_json_data(json_dir, kimg, fid):
    if not os.path.exists(json_dir):
        with open(json_dir, 'w') as f:
            json.dump({"fid_evaluation": []}, f, indent=4)

    with open(json_dir, 'r') as f:
        data = json.load(f)
    
    new_entry = {"kimg": kimg, "fid": fid}
    data["fid_evaluation"].append(new_entry)

    with open(json_dir, 'w') as f:
        json.dump(data, f, indent=4)

def fid_score(kimg, dir, generator, latent_dim, real_dir, device, n_samples=50000, batch_size=100):
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp_fake_dir:
        generate_fake_images(generator, latent_dim, tmp_fake_dir, n_samples, batch_size, device)
        
        score = fid_score_func.calculate_fid_given_paths(
            [real_dir, tmp_fake_dir],
            batch_size=batch_size,
            device=device,
            dims=2048,
        )

        out = os.path.join(dir, 'FID_Evaluation.json')
        
        dump_json_data(out, kimg, score)

        return score

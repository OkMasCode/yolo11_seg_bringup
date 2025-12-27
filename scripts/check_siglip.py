
import torch
import open_clip
import numpy as np

def check_siglip():
    model_name = 'ViT-B-16-SigLIP'
    pretrained = 'webli'
    device = 'cpu'
    
    print(f"Creating model {model_name}...")
    model, _, _ = open_clip.create_model_and_transforms(
        model_name, 
        pretrained=pretrained,
        device=device
    )
    
    print("Model created.")
    
    if hasattr(model, 'logit_scale'):
        scale = model.logit_scale
        print(f"logit_scale (raw): {scale.item()}")
        print(f"logit_scale (exp): {scale.exp().item()}")
    else:
        print("logit_scale not found")
        
    if hasattr(model, 'logit_bias'):
        bias = model.logit_bias
        print(f"logit_bias: {bias.item()}")
    else:
        print("logit_bias not found")

    # Test computation
    # Create dummy embeddings
    img_emb = np.random.rand(768)
    img_emb /= np.linalg.norm(img_emb)
    
    txt_emb = np.random.rand(768)
    txt_emb /= np.linalg.norm(txt_emb)
    
    dot = np.dot(img_emb, txt_emb)
    print(f"Dummy dot product: {dot}")
    
    if hasattr(model, 'logit_scale') and hasattr(model, 'logit_bias'):
        logit_scale = model.logit_scale.exp().item()
        logit_bias = model.logit_bias.item()
        
        logits = (dot * logit_scale) + logit_bias
        probs = 1 / (1 + np.exp(-logits))
        print(f"Computed prob: {probs}")

if __name__ == "__main__":
    check_siglip()

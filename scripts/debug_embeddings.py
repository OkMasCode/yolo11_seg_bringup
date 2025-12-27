import numpy as np
import torch
import open_clip

def test_embedding_computation():
    """Debug script to test embedding shapes and similarity computation"""
    
    model_name = 'ViT-B-16-SigLIP'
    device = 'cpu'
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, 
        pretrained='webli',
        device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    
    # Test text embedding
    text_prompt = "a person sitting"
    tokens = tokenizer([text_prompt]).to(device)
    
    with torch.no_grad():
        text_feat = model.encode_text(tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        text_emb = text_feat.squeeze(0).detach().cpu().numpy()
    
    print(f"Text embedding shape: {text_emb.shape}")
    print(f"Text embedding dtype: {text_emb.dtype}")
    print(f"Text embedding norm: {np.linalg.norm(text_emb)}")
    print(f"Text embedding sample values: {text_emb[:5]}")
    print()
    
    # Simulate image embedding
    image_emb = np.random.randn(768).astype(np.float32)
    image_emb = image_emb / np.linalg.norm(image_emb)
    
    print(f"Image embedding shape: {image_emb.shape}")
    print(f"Image embedding dtype: {image_emb.dtype}")
    print(f"Image embedding norm: {np.linalg.norm(image_emb)}")
    print()
    
    # Test dot product
    dot_product = np.dot(image_emb, text_emb)
    print(f"Dot product: {dot_product}")
    print(f"Dot product type: {type(dot_product)}")
    print()
    
    # Get logit scale and bias
    logit_scale = model.logit_scale.exp().item()
    logit_bias = model.logit_bias.item()
    
    print(f"Logit scale: {logit_scale}")
    print(f"Logit bias: {logit_bias}")
    print()
    
    # Compute probability
    logits = (dot_product * logit_scale) + logit_bias
    probs = 1 / (1 + np.exp(-logits))
    
    print(f"Logits: {logits}")
    print(f"Probs: {probs}")
    print(f"Probs as percentage: {probs*100:.1f}%")

if __name__ == "__main__":
    test_embedding_computation()

import numpy as np
import torch
import open_clip

def test_matching_embeddings():
    """Test similarity with actual matching text and images"""
    
    model_name = 'ViT-B-16-SigLIP'
    device = 'cpu'
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, 
        pretrained='webli',
        device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    
    # Test 1: Same text twice (should be high similarity)
    text1 = "a person sitting on a chair"
    text2 = "a person sitting on a chair"
    
    tokens1 = tokenizer([text1]).to(device)
    tokens2 = tokenizer([text2]).to(device)
    
    with torch.no_grad():
        feat1 = model.encode_text(tokens1)
        feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
        
        feat2 = model.encode_text(tokens2)
        feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)
    
    dot = np.dot(feat1.cpu().numpy().flatten(), feat2.cpu().numpy().flatten())
    logit_scale = model.logit_scale.exp().item()
    logit_bias = model.logit_bias.item()
    logits = (dot * logit_scale) + logit_bias
    probs = 1 / (1 + np.exp(-logits))
    
    print("Test 1: Identical text")
    print(f"  Dot product: {dot:.4f}")
    print(f"  Similarity: {probs*100:.1f}%")
    print()
    
    # Test 2: Related text
    text1 = "a person"
    text2 = "a person sitting"
    
    tokens1 = tokenizer([text1]).to(device)
    tokens2 = tokenizer([text2]).to(device)
    
    with torch.no_grad():
        feat1 = model.encode_text(tokens1)
        feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
        
        feat2 = model.encode_text(tokens2)
        feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)
    
    dot = np.dot(feat1.cpu().numpy().flatten(), feat2.cpu().numpy().flatten())
    logits = (dot * logit_scale) + logit_bias
    probs = 1 / (1 + np.exp(-logits))
    
    print("Test 2: Related text")
    print(f"  Dot product: {dot:.4f}")
    print(f"  Similarity: {probs*100:.1f}%")
    print()
    
    # Test 3: Unrelated text
    text1 = "a dog"
    text2 = "a person"
    
    tokens1 = tokenizer([text1]).to(device)
    tokens2 = tokenizer([text2]).to(device)
    
    with torch.no_grad():
        feat1 = model.encode_text(tokens1)
        feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
        
        feat2 = model.encode_text(tokens2)
        feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)
    
    dot = np.dot(feat1.cpu().numpy().flatten(), feat2.cpu().numpy().flatten())
    logits = (dot * logit_scale) + logit_bias
    probs = 1 / (1 + np.exp(-logits))
    
    print("Test 3: Unrelated text")
    print(f"  Dot product: {dot:.4f}")
    print(f"  Similarity: {probs*100:.1f}%")

if __name__ == "__main__":
    test_matching_embeddings()

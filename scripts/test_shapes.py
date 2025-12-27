import numpy as np

# Simulate what happens in the code
emb1 = np.random.randn(1, 768)  # What if encode_images_batch returns (1, 768)?
emb2 = np.random.randn(768)      # What encode_text returns

print(f"Image embedding shape: {emb1.shape}")
print(f"Text embedding shape: {emb2.shape}")

# Normalize
emb1_norm = emb1 / np.linalg.norm(emb1)
emb2_norm = emb2 / np.linalg.norm(emb2)

print(f"Normalized image shape: {emb1_norm.shape}")
print(f"Normalized text shape: {emb2_norm.shape}")

# Try dot product
try:
    dot1 = np.dot(emb1_norm, emb2_norm)
    print(f"Dot product (1,768) x (768): {dot1}")
except Exception as e:
    print(f"Error with (1,768) x (768): {e}")

try:
    dot2 = np.dot(emb1_norm.flatten(), emb2_norm)
    print(f"Dot product (768,) x (768): {dot2}")
except Exception as e:
    print(f"Error: {e}")

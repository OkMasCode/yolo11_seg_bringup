#!/usr/bin/env python3
"""CLIP embedding processing utilities."""
import torch
import clip
import cv2
from PIL import Image as PILImage


class CLIPProcessor:
    """Handles CLIP model loading and embedding generation."""
    
    def __init__(self, device="cuda", model_name="ViT-B/32"):
        """
        Initialize CLIP processor.
        
        Args:
            device: torch device (cuda/cpu)
            model_name: CLIP model variant
        """
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
    
    def encode_text_prompt(self, text):
        """
        Encode text prompt to CLIP embedding.
        
        Args:
            text: Text string to encode
            
        Returns:
            Normalized text features tensor
        """
        with torch.no_grad():
            text_token = clip.tokenize([text]).to(self.device)
            text_features = self.model.encode_text(text_token)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def encode_image_crop(self, crop_bgr):
        """
        Encode image crop to CLIP embedding.
        
        Args:
            crop_bgr: BGR image crop (numpy array)
            
        Returns:
            Normalized image features as numpy array
        """
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_crop = PILImage.fromarray(crop_rgb)
        image_in = self.preprocess(pil_crop).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feat = self.model.encode_image(image_in)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        
        return feat.squeeze(0).detach().float().cpu().numpy()
    
    @staticmethod
    def compute_square_crop(x1, y1, x2, y2, width, height, scale=1.4):
        """
        Compute square crop coordinates around bounding box.
        
        Args:
            x1, y1, x2, y2: Bounding box coordinates
            width, height: Image dimensions
            scale: Scale factor for crop (default 1.4 = 40% larger)
            
        Returns:
            tuple: (sx1, sy1, sx2, sy2) square crop coordinates
        """
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        side = max(bw, bh)
        side = int(round(side * max(1.3, float(scale))))
        
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        half = side / 2.0
        
        sx1 = int(round(cx - half))
        sy1 = int(round(cy - half))
        sx2 = int(round(cx + half))
        sy2 = int(round(cy + half))
        
        # Clamp to image bounds
        sx1 = max(0, min(sx1, width - 1))
        sy1 = max(0, min(sy1, height - 1))
        sx2 = max(0, min(sx2, width - 1))
        sy2 = max(0, min(sy2, height - 1))
        
        return sx1, sy1, sx2, sy2

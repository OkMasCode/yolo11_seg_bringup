import torch
import cv2
import numpy as np

import open_clip
from PIL import Image as PILImage

class CLIPProcessor:
    """
    Handles SigLIP model processing tasks:
    - Square image cropping
    - Image preprocessing (CLAHE, saturation adjustment)
    - Ecoding text and images for model input
    - Computing sigmoid probabilities
    """
    def __init__(self, device='cpu', model_name='ViT-B-16-SigLIP', pretrained='webli'):
        """
        Iniialize CLIPProcessor
        """
        self.device = device
        print(f"Loading SigLIP model: {model_name} ({pretrained})...")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained,
            device=device
        )

        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model.eval()

    @staticmethod
    def compute_square_crop(x1,y1,x2,y2, width, height, scale = 1.2):
        """
        Compute square crop coordinates
        """
        bw, bh = x2 - x1, y2 - y1
        side = int(max(bw, bh) * scale)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        
        sx1 = max(0, int(cx - side / 2))
        sy1 = max(0, int(cy - side / 2))
        sx2 = min(width, int(cx + side / 2))
        sy2 = min(height, int(cy + side / 2))
        return sx1, sy1, sx2, sy2
        
    def encode_text(self, text):
        """
        Encode text prompts to a single normalized SigLIP embedding.
        Supports prompt ensembling by averaging multiple prompts.
        """
        if text is None: return None

        if isinstance(text, str):
            if not text.strip():
                return None
            text = [text]
        elif isinstance(text, list):
            text = [t for t in text if isinstance(t, str) and t.strip()]
            if not text:  
                return None
        
        with torch.no_grad():
            tokens = self.tokenizer(text).to(self.device)

            feats = self.model.encode_text(tokens)

            feats = feats / feats.norm(dim=-1, keepdim=True)

            ensamble_feat = feats.mean(dim=0, keepdim=True)
            ensamble_feat = ensamble_feat / ensamble_feat.norm(dim=-1, keepdim=True)

        return ensamble_feat.squeeze(0).detach().cpu().numpy()

    def preproc_image(self, image_bgr):
        """
        Applies Contrast Limited Adaptive Histogram Equaization (CLAHE)
        and saturation adjustment.
        """
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        contrast_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        hsv = cv2.cvtColor(contrast_enhanced, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(s * 1.25, 0, 255).astype(np.uint8)
        hsv_boosted = cv2.merge([h, s, v])

        final = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)
        return final

    def encode_images_batch(self, crops_bgr):
        """
        Encode a batch of image crops to SigLIP embeddings.
        """
        if not crops_bgr: return []

        processed_crops = []
        for c in crops_bgr:
            enhanced_c = self.preproc_image(c)
            rgb_c = cv2.cvtColor(enhanced_c, cv2.COLOR_BGR2RGB)
            processed_crops.append(PILImage.fromarray(rgb_c))

        image_input = torch.stack([self.preprocess(img) for img in processed_crops]).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu().numpy()
    
    def compute_sigmoid_probs(self, image_embedding, text_embedding):
        """
        Converts raw SigLIP dot product into a readable probability.
        Formula: sigmoid( dot_product * scale + bias)
        
        Returns:
            tuple: (dot_product, probability) or (None, None) if embeddings are None
        """
        if image_embedding is None or text_embedding is None:
            return None

        img = np.array(image_embedding).flatten()
        txt = np.array(text_embedding).flatten()
        
        if img.size != txt.size:
            raise ValueError(f"Embedding size mismatch: image {img.size} vs text {txt.size}")
        
        dot_product = np.dot(img, txt)
        
        with torch.no_grad():
            logit_scale = self.model.logit_scale.exp().item()
            logit_bias = self.model.logit_bias.item()
        
        logits = (dot_product * logit_scale) + logit_bias
        probs = 1 / (1 + np.exp(-logits))
        return float(probs)

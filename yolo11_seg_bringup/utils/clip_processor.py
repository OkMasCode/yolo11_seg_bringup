import torch
import cv2
import numpy as np
from time import perf_counter

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
    def __init__(self, device='cpu', model_name='ViT-B-16-SigLIP', pretrained='webli', image_size=224):
        """
        Iniialize CLIPProcessor
        """
        self.device = device
        self.model_name = model_name
        self.image_size = image_size
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

    def _batch_preprocess_opencv(self, crops_bgr):
        """
        Vectorized batch preprocessing using OpenCV + NumPy.
        Replaces PIL-based sequential preprocessing for massive speedup.
        
        Args:
            crops_bgr: List of BGR images (NumPy arrays)
            
        Returns:
            Tensor of shape (N, 3, H, W) ready for model input
        """
        if not crops_bgr:
            return torch.empty(0, 3, self.image_size, self.image_size, device=self.device)
        
        # ===== Batch Resize (OpenCV) =====
        resize_start = perf_counter()
        resized = []
        for crop in crops_bgr:
            resized_img = cv2.resize(crop, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            resized.append(resized_img)
        resize_time = (perf_counter() - resize_start) * 1000
        print(f"[CLIP] Batch resize to {self.image_size}x{self.image_size}: {resize_time:.2f}ms")
        
        # ===== Stack into NumPy batch =====
        stack_start = perf_counter()
        batch_bgr = np.stack(resized, axis=0).astype(np.float32)  # (N, H, W, 3)
        stack_time = (perf_counter() - stack_start) * 1000
        print(f"[CLIP] Stack batch: {stack_time:.2f}ms")
        
        # ===== BGR to RGB (vectorized) =====
        convert_start = perf_counter()
        batch_rgb = batch_bgr[..., ::-1]  # Flip RGB channels
        convert_time = (perf_counter() - convert_start) * 1000
        print(f"[CLIP] BGR→RGB conversion: {convert_time:.2f}ms")
        
        # ===== Normalize to [0, 1] =====
        normalize_01_start = perf_counter()
        batch_rgb = batch_rgb / 255.0
        normalize_01_time = (perf_counter() - normalize_01_start) * 1000
        print(f"[CLIP] Normalize to [0, 1]: {normalize_01_time:.2f}ms")
        
        # ===== ImageNet normalization (vectorized) =====
        im_norm_start = perf_counter()
        # Standard ImageNet mean/std used by open_clip
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        batch_rgb = (batch_rgb - mean) / std
        im_norm_time = (perf_counter() - im_norm_start) * 1000
        print(f"[CLIP] ImageNet normalization: {im_norm_time:.2f}ms")
        
        # ===== Convert to tensor and rearrange to (N, 3, H, W) =====
        tensor_start = perf_counter()
        batch_tensor = torch.from_numpy(batch_rgb).permute(0, 3, 1, 2)
        tensor_time = (perf_counter() - tensor_start) * 1000
        print(f"[CLIP] NumPy→Tensor+Permute: {tensor_time:.2f}ms")
        
        return batch_tensor

    def encode_images_batch(self, crops_bgr):
        """
        Encode a batch of image crops to SigLIP embeddings.
        """
        if not crops_bgr: return []
        
        total_start = perf_counter()

        # ===== Fast Vectorized Preprocessing =====
        image_input = self._batch_preprocess_opencv(crops_bgr)

        # ===== Move to GPU =====
        to_device_start = perf_counter()
        image_input = image_input.to(self.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        to_device_time = (perf_counter() - to_device_start) * 1000
        print(f"[CLIP] Transfer to {self.device}: {to_device_time:.2f}ms")

        # ===== Model inference =====
        inference_start = perf_counter()
        with torch.no_grad():
            features = self.model.encode_image(image_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        inference_time = (perf_counter() - inference_start) * 1000
        print(f"[CLIP] Model inference: {inference_time:.2f}ms")

        # ===== Normalization =====
        norm_start = perf_counter()
        features = features / features.norm(dim=-1, keepdim=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        norm_time = (perf_counter() - norm_start) * 1000
        print(f"[CLIP] Normalization: {norm_time:.2f}ms")

        # ===== Transfer to CPU =====
        cpu_start = perf_counter()
        result = features.cpu().numpy()
        cpu_time = (perf_counter() - cpu_start) * 1000
        print(f"[CLIP] Transfer to CPU: {cpu_time:.2f}ms")
        
        total_time = (perf_counter() - total_start) * 1000
        print(f"[CLIP] === TOTAL: {total_time:.2f}ms ===\n")
        
        return result
    
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
        probs = float(1.0 / (1.0 + np.exp(-logits))*100.0)
        return float(probs)

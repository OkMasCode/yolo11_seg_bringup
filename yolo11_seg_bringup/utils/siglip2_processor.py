import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from transformers import AutoProcessor, AutoModel

class CLIPProcessorValidator:
    """
    SOTA Hugging Face SigLIP 2 Implementation
    Features:
    - Native Aspect Ratio Handling (No more forced square padding)
    - SigLIP 2 Large backbone for superior dense localization
    - Prompt/text embedding with optional prompt ensembling
    - Sigmoid-based similarity score with configurable masked/unmasked blending
    """

    def __init__(
        self,
        device="cuda",
        model_name="google/siglip2-large-patch16-384",
        masked_score_weight=0.85,
        unmasked_score_weight=0.15,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name

        total_weight = float(masked_score_weight) + float(unmasked_score_weight)
        if total_weight <= 0.0:
            self.masked_score_weight = 0.5
            self.unmasked_score_weight = 0.5
        else:
            self.masked_score_weight = float(masked_score_weight) / total_weight
            self.unmasked_score_weight = float(unmasked_score_weight) / total_weight

        # ---------------------------------------------------------
        # SOTA CHANGE: Use Hugging Face AutoClasses instead of open_clip
        # ---------------------------------------------------------
        print(f"[CLIPProcessor] Loading SOTA model: {self.model_name} on {self.device}")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        with torch.no_grad():
            self.cached_logit_scale = (
                float(self.model.logit_scale.exp().item())
                if hasattr(self.model, "logit_scale")
                else 1.0
            )
            self.cached_logit_bias = (
                float(self.model.logit_bias.item())
                if hasattr(self.model, "logit_bias")
                else -10.0 # Safe default for SigLIP if bias is missing
            )

    @staticmethod
    def build_prompt_list(label: str) -> list:
        clean_label = label.strip()
        return [
            f"a photo of a {clean_label}.",
            f"a bad photo of a {clean_label}.",
            f"a photo of the large {clean_label}.",
            f"a photo of the small {clean_label}.",
            f"a cropped photo of a {clean_label}.",
            f"a close-up photo of a {clean_label}.",
            f"a clear photo of a {clean_label}.",
        ]

    def encode_text(self, text):
        if text is None:
            return None

        if isinstance(text, str):
            if not text.strip(): return None
            text_list = [text.strip()]
        elif isinstance(text, list):
            text_list = [t.strip() for t in text if isinstance(t, str) and t.strip()]
            if not text_list: return None
        else:
            return None

        with torch.no_grad():
            inputs = self.processor(text=text_list, padding="max_length", return_tensors="pt").to(self.device)
            
            # Use dedicated text path to avoid joint loss requirement
            outputs = self.model.get_text_features(**inputs)
            
            # SOTA EDGE FIX: Safely extract the tensor from the HF object.
            # SigLIP uses attention pooling, so the final embedding is the pooler_output.
            text_features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
            
            # Normalize and ensemble
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            ensemble_feature = text_features.mean(dim=0, keepdim=True)
            ensemble_feature = ensemble_feature / ensemble_feature.norm(p=2, dim=-1, keepdim=True)

        return ensemble_feature.squeeze(0).detach().cpu().numpy() 

    def encode_images_batch(self, images_bgr: list):
        if not images_bgr:
            return []

        pil_images = []
        for image_bgr in images_bgr:
            if image_bgr is None or image_bgr.size == 0:
                continue
            
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_images.append(PILImage.fromarray(image_rgb))

        if not pil_images:
            return []

        with torch.no_grad():
            inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
            
            # Use dedicated vision path to avoid joint loss requirement
            outputs = self.model.get_image_features(**inputs)
            
            # SOTA EDGE FIX: Safely extract the tensor from the HF object.
            image_features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
            
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        return [embedding for embedding in image_features.detach().cpu().numpy()]

    def prepare_crops(self, cv_bgr, mask_uint8, bbox_xyxy):
        """
        SOTA CHANGE: We simply extract the tight bounding box.
        SigLIP 2 handles the native aspect ratio via its processor.
        """
        height, width = cv_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

        if x2 <= x1 or y2 <= y1:
            return None, None

        img_crop = cv_bgr[y1:y2, x1:x2]
        unmasked_crop = img_crop.copy()
        mask_crop = mask_uint8[y1:y2, x1:x2]

        if img_crop.size == 0 or mask_crop.size == 0:
            return None, None

        # Dilate mask slightly to capture object edges
        mask_crop = cv2.dilate(mask_crop, np.ones((5, 5), np.uint8), iterations=1)
        neutral_bg = np.full_like(img_crop, 122)

        bg_mask = cv2.bitwise_not(mask_crop)
        fg = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)
        bg = cv2.bitwise_and(neutral_bg, neutral_bg, mask=bg_mask)
        masked_crop = cv2.add(fg, bg)
        
        return masked_crop, unmasked_crop

    @staticmethod
    def _safe_sigmoid(value: float) -> float:
        clipped = np.clip(value, -60.0, 60.0)
        return float(1.0 / (1.0 + np.exp(-clipped)))

    def compute_match_score(self, image_embedding, text_embedding) -> float:
        image_vec = np.asarray(image_embedding, dtype=np.float32).flatten()
        text_vec = np.asarray(text_embedding, dtype=np.float32).flatten()

        dot_product = float(np.dot(image_vec, text_vec))
        logits = (dot_product * self.cached_logit_scale) + self.cached_logit_bias
        return self._safe_sigmoid(logits) * 100.0

    def compute_blended_match_score(
        self,
        masked_embedding,
        unmasked_embedding,
        text_embedding,
    ) -> float:
        masked_score = self.compute_match_score(masked_embedding, text_embedding)
        unmasked_score = self.compute_match_score(unmasked_embedding, text_embedding)
        return (
            (self.masked_score_weight * masked_score)
            + (self.unmasked_score_weight * unmasked_score)
        )

    def encode_label_prompt(self, label: str):
        return self.encode_text(self.build_prompt_list(label))
import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from transformers import AutoProcessor, AutoModel

class SIGLIPProcessor:
    """
    SOTA Hugging Face SigLIP 2 Implementation
    Features:
    - Native Aspect Ratio Handling
    - SigLIP 2 Base backbone for faster dense localization on edge devices
    - Prompt/text embedding with optional prompt ensembling
    - Sigmoid-based similarity score with configurable masked/unmasked blending
    - Patch-level extraction for dense heatmaps (ROS 2 compatible)
    """

    def __init__(
        self,
        device="cuda",
        # Default changed to the Base model for better FPS on Jetson
        model_name="/home/workspace/local_siglip_model_base_256",
        masked_score_weight=0.85,
        unmasked_score_weight=0.15,
        # Added offline flag for Jetson deployment
        offline_mode=False 
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
            
        print(f"[CLIPProcessor] Loading model: {self.model_name} on {self.device}")
        
        # Pass local_files_only to prevent internet pings if offline_mode is True
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, 
            local_files_only=offline_mode
        )
        self.model = AutoModel.from_pretrained(
            self.model_name, 
            local_files_only=offline_mode
        ).to(self.device)
        
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
            outputs = self.model.get_text_features(**inputs)
            text_features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
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
            outputs = self.model.get_image_features(**inputs)
            image_features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return [embedding for embedding in image_features.detach().cpu().numpy()]

    def encode_image(self, image_bgr):
        """Encode one BGR image and return a single embedding vector."""
        embeddings = self.encode_images_batch([image_bgr])
        if not embeddings:
            return None
        return embeddings[0]

    def encode_image_patches_batch(self, images_bgr: list):
        """
        Extracts the raw patch embeddings for a batch of images before they are pooled.
        Returns a list of numpy arrays, where each array contains the grid of patches.
        """
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
            # Bypass the pooler and extract the raw patches
            outputs = self.model.vision_model(**inputs)
            patch_features = outputs.last_hidden_state
            
            # Normalize patches for cosine similarity matching
            patch_features = patch_features / patch_features.norm(p=2, dim=-1, keepdim=True)
            
        return [patches for patches in patch_features.detach().cpu().numpy()]

    def encode_image_patches(self, image_bgr):
        """Encode one BGR image and return its 2D grid of patch embeddings."""
        patch_embeddings = self.encode_image_patches_batch([image_bgr])
        if not patch_embeddings:
            return None
        return patch_embeddings[0]

    def get_dense_feature_map(self, image_bgr):
        """
        Extracts the spatial 2D feature map.
        Returns: numpy array of shape (grid_h, grid_w, hidden_dim)
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(image_rgb)
        
        with torch.no_grad():
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
            # Find the resolution the processor resized the image to (e.g., 384x384)
            _, _, h_res, w_res = inputs['pixel_values'].shape
            
            outputs = self.model.vision_model(**inputs)
            patch_features = outputs.last_hidden_state # [1, Num_Patches, Dim]
            
            # Calculate grid dimensions based on the patch size
            patch_size = self.model.config.vision_config.patch_size
            grid_h = h_res // patch_size
            grid_w = w_res // patch_size
            
            # Note: Standard OpenAI CLIP has a CLS token at index 0.
            # If you switch from SigLIP to standard CLIP, you must slice it out:
            # patch_features = patch_features[:, 1:, :]
            
            dim = patch_features.shape[-1]
            
            # Reshape from [1, grid_h * grid_w, dim] -> [grid_h, grid_w, dim]
            spatial_features = patch_features.view(grid_h, grid_w, dim)
            
            # L2 Normalize along the channel dimension (Matches FindAnything C++ line 249)
            spatial_features = spatial_features / spatial_features.norm(p=2, dim=-1, keepdim=True)
            
        return spatial_features.detach().cpu().numpy()
    
    def generate_findanything_heatmap(self, image_bgr, text_prompt):
        """
        Replicates the exact heatmap generation logic from FindAnything/OKVIS2-X.
        """
        # 1. Get Text Embedding (Normalized)
        text_embedding = self.encode_label_prompt(text_prompt) 
        
        # 2. Get Coarse 2D Feature Map [grid_h, grid_w, dim]
        feature_map = self.get_dense_feature_map(image_bgr) 
        
        # 3. Compute Activation Map (Dot Product)
        # Because both are L2 normalized, this equates to Cosine Similarity
        activation_map = np.dot(feature_map, text_embedding) 
        
        # 4. Clamp & Scale (Matches C++: activationMap.clamp(0.0f, 1.0f) * 254.0f)
        activation_map = np.clip(activation_map, 0.0, 1.0)
        vis_feature_map = (activation_map * 254.0).astype(np.uint8)
        
        # 5. Apply Colormap (Matches C++: cv::COLORMAP_JET)
        vis_colored = cv2.applyColorMap(vis_feature_map, cv2.COLORMAP_JET)
        
        # 6. Upscale to original RGB resolution (Matches C++: cv::INTER_CUBIC)
        h_orig, w_orig = image_bgr.shape[:2]
        vis_colored_resized = cv2.resize(vis_colored, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
        
        # 7. Blend with the original image (Matches C++: cv::addWeighted)
        blended = cv2.addWeighted(image_bgr, 0.2, vis_colored_resized, 0.8, 0.0)
        
        return blended, activation_map
    
    def prepare_crops(self, cv_bgr, mask_uint8, bbox_xyxy):
        height, width = cv_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return None, None
        img_crop = cv_bgr[y1:y2, x1:x2]
        unmasked_crop = img_crop.copy()
        mask_crop = mask_uint8[y1:y2, x1:x2]
        if img_crop.size == 0 or mask_crop.size == 0:
            return None, None
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

    def compute_match_logit(self, image_embedding, text_embedding) -> float:
        """Return the raw SigLIP logit before sigmoid conversion."""
        image_vec = np.asarray(image_embedding, dtype=np.float64).flatten()
        text_vec = np.asarray(text_embedding, dtype=np.float64).flatten()
        dot_product = float(np.dot(image_vec, text_vec))
        logits = (dot_product * float(self.cached_logit_scale)) + float(self.cached_logit_bias)
        return float(np.float64(logits))

    def compute_match_score_raw(self, image_embedding, text_embedding) -> float:
        return self.compute_match_logit(image_embedding, text_embedding)

    def compute_match_score(self, image_embedding, text_embedding) -> float:
        logits = self.compute_match_logit(image_embedding, text_embedding)
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
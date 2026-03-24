import cv2
import numpy as np
import open_clip
import torch
from PIL import Image as PILImage


class CLIPProcessorValidator:
    """
    CLIP/SigLIP processor that mirrors the logic used in clip_model_validator_node.

    Features:
    - Square crop computation shared with detector bounding boxes
    - Masked and unmasked crop preparation
    - Prompt/text embedding with optional prompt ensembling
    - Batch image embedding with validator-style square padding
    - Sigmoid-based similarity score with configurable masked/unmasked blending
    """

    def __init__(
        self,
        device="cpu",
        model_name="ViT-B-16-SigLIP",
        pretrained="webli",
        square_crop_scale=1.2,
        masked_score_weight=0.5,
        unmasked_score_weight=0.5,
    ):
        self.device = device
        self.model_name = model_name
        self.square_crop_scale = float(square_crop_scale)

        total_weight = float(masked_score_weight) + float(unmasked_score_weight)
        if total_weight <= 0.0:
            self.masked_score_weight = 0.5
            self.unmasked_score_weight = 0.5
        else:
            self.masked_score_weight = float(masked_score_weight) / total_weight
            self.unmasked_score_weight = float(unmasked_score_weight) / total_weight

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=self.model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
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
                else 0.0
            )

    @staticmethod
    def _pad_to_square(image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr is None or image_bgr.size == 0:
            return image_bgr

        h, w = image_bgr.shape[:2]
        if h == w:
            return image_bgr

        side = max(h, w)
        canvas = np.full((side, side, 3), 128, dtype=np.uint8)
        x_offset = (side - w) // 2
        y_offset = (side - h) // 2
        canvas[y_offset : y_offset + h, x_offset : x_offset + w] = image_bgr
        return canvas

    @staticmethod
    def compute_square_crop(x1, y1, x2, y2, width, height, scale=1.2):
        bw = x2 - x1
        bh = y2 - y1
        side = int(max(bw, bh) * scale)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        sx1 = max(0, int(cx - side / 2))
        sy1 = max(0, int(cy - side / 2))
        sx2 = min(width, int(cx + side / 2))
        sy2 = min(height, int(cy + side / 2))
        return sx1, sy1, sx2, sy2

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
        """
        Encode text prompts into one normalized embedding.

        `text` can be:
        - str: a single prompt
        - list[str]: prompt ensemble
        """
        if text is None:
            return None

        if isinstance(text, str):
            if not text.strip():
                return None
            text_list = [text.strip()]
        elif isinstance(text, list):
            text_list = [t.strip() for t in text if isinstance(t, str) and t.strip()]
            if not text_list:
                return None
        else:
            return None

        with torch.no_grad():
            try:
                tokens = self.tokenizer(text_list).to(self.device)
            except AttributeError:
                if hasattr(self.tokenizer, "tokenizer"):
                    hf_tokenizer = self.tokenizer.tokenizer
                    context_length = getattr(self.tokenizer, "context_length", 77)
                    encoded = hf_tokenizer(
                        text_list,
                        padding="max_length",
                        truncation=True,
                        max_length=context_length,
                        return_tensors="pt",
                    )
                    tokens = encoded["input_ids"].to(self.device)
                else:
                    raise

            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            ensemble_feature = features.mean(dim=0, keepdim=True)
            ensemble_feature = ensemble_feature / ensemble_feature.norm(
                dim=-1,
                keepdim=True,
            )

        return ensemble_feature.squeeze(0).detach().cpu().numpy()

    def encode_images_batch(self, images_bgr: list):
        if not images_bgr:
            return []

        image_tensors = []
        for image_bgr in images_bgr:
            if image_bgr is None or image_bgr.size == 0:
                continue
            squared_bgr = self._pad_to_square(image_bgr)
            image_rgb = cv2.cvtColor(squared_bgr, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(image_rgb)
            image_tensors.append(self.preprocess(pil_image))

        if not image_tensors:
            return []

        with torch.no_grad():
            image_input = torch.stack(image_tensors, dim=0).to(self.device)
            features = self.model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)

        return [embedding for embedding in features.detach().cpu().numpy()]

    def prepare_crops(self, cv_bgr, mask_uint8, bbox_xyxy):
        """
        Build the same pair used by the validator:
        1) masked crop with neutral gray background outside the object mask
        2) unmasked crop from the original image
        """
        height, width = cv_bgr.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy

        sx1, sy1, sx2, sy2 = self.compute_square_crop(
            x1,
            y1,
            x2,
            y2,
            width,
            height,
            self.square_crop_scale,
        )
        if sx2 <= sx1 or sy2 <= sy1:
            return None, None

        img_crop = cv_bgr[sy1:sy2, sx1:sx2]
        unmasked_crop = img_crop.copy()
        mask_crop = mask_uint8[sy1:sy2, sx1:sx2]

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
        """Helper to encode validator-style prompt templates for one label."""
        return self.encode_text(self.build_prompt_list(label))

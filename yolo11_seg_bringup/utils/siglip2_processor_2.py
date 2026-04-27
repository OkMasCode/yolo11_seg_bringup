import cv2
import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from transformers import AutoProcessor, AutoModel

class SIGLIPProcessor:
    def __init__(
        self,
        engine_path="/home/workspace/siglip_vision_pooled_384_fp16.engine",
        model_name="/home/workspace/local_siglip_model",
        masked_score_weight=0.85,
        unmasked_score_weight=0.15,
        max_batch_size=32 # Must match the --maxShapes used in trtexec
    ):
        self.model_name = model_name
        self.max_batch_size = max_batch_size
        
        # --- Weight Calculation ---
        total_weight = float(masked_score_weight) + float(unmasked_score_weight)
        if total_weight <= 0.0:
            self.masked_score_weight = 0.5
            self.unmasked_score_weight = 0.5
        else:
            self.masked_score_weight = float(masked_score_weight) / total_weight
            self.unmasked_score_weight = float(unmasked_score_weight) / total_weight
            
        print(f"[SIGLIPProcessor] Loading TRT Engine: {engine_path}")

        # --- 1. Load TensorRT Vision Engine ---
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Allocate TRT Buffers
        self.inputs = {}
        self.outputs = {}
        self.allocations = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            
            # Use max batch size for allocation to prevent reallocation during runtime
            shape = list(self.engine.get_tensor_shape(name))
            if shape[0] == -1: 
                shape[0] = self.max_batch_size 
            
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = np.abs(trt.volume(shape)) * np.dtype(dtype).itemsize
            
            host_mem = cuda.pagelocked_empty(shape, dtype)
            device_mem = cuda.mem_alloc(int(size))
            
            self.allocations.append(device_mem)
            self.context.set_tensor_address(name, int(device_mem))
            
            if is_input:
                self.inputs[name] = {'host': host_mem, 'device': device_mem, 'shape': shape}
            else:
                self.outputs[name] = {'host': host_mem, 'device': device_mem, 'shape': shape}

        # Pre-compute SigLIP preprocessing constants
        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)

        # --- 2. Load PyTorch Text Encoder ---
        print(f"[SIGLIPProcessor] Loading PyTorch Text Encoder: {self.model_name}")
        self.processor = AutoProcessor.from_pretrained(self.model_name, local_files_only=True)
        self.text_model = AutoModel.from_pretrained(self.model_name, local_files_only=True).to("cuda").eval()
        
        # Cache Logit Scale/Bias
        with torch.no_grad():
            self.cached_logit_scale = float(self.text_model.logit_scale.exp().item()) if hasattr(self.text_model, "logit_scale") else 1.0
            self.cached_logit_bias = float(self.text_model.logit_bias.item()) if hasattr(self.text_model, "logit_bias") else -10.0

    # --- TEXT ENCODING (PyTorch) ---
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
        if not text: return None
        text_list = [t.strip() for t in (text if isinstance(text, list) else [text]) if t.strip()]
        if not text_list: return None
        
        with torch.no_grad():
            inputs = self.processor(text=text_list, padding="max_length", return_tensors="pt").to("cuda")
            outputs = self.text_model.get_text_features(**inputs)
            
            # Handle standard HuggingFace outputs
            text_features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            
            ensemble_feature = text_features.mean(dim=0, keepdim=True)
            ensemble_feature = ensemble_feature / ensemble_feature.norm(p=2, dim=-1, keepdim=True)
            
        return ensemble_feature.squeeze(0).cpu().numpy() 

    def encode_label_prompt(self, label: str):
        return self.encode_text(self.build_prompt_list(label))

    def _preprocess_images(self, images_bgr: list):
        """Vectorized NumPy batch preprocessing."""
        batch_resized = []
        for img in images_bgr:
            if img is None or img.size == 0: continue
            
            # 1. Resize and color convert (Keep as uint8 for now)
            img_resized = cv2.resize(img, (384, 384), interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            batch_resized.append(img_rgb)
            
        if not batch_resized: return None
        
        # 2. Stack into a single continuous memory block (Fast)
        stacked_uint8 = np.stack(batch_resized) # Shape: (B, 384, 384, 3)
        
        # 3. Vectorized Math: Apply float conversion and normalization to the whole batch at once
        stacked_float = stacked_uint8.astype(np.float32) / 255.0
        stacked_normalized = (stacked_float - self.mean) / self.std
        
        # 4. Transpose the entire batch: (B, H, W, C) -> (B, C, H, W)
        return np.transpose(stacked_normalized, (0, 3, 1, 2))

    def encode_images_batch(self, images_bgr: list):
        """Encodes a batch of images (e.g., YOLO crops or full scenes) using TensorRT."""
        input_data = self._preprocess_images(images_bgr)
        if input_data is None: return []
        
        batch_size = input_data.shape[0]
        if batch_size > self.max_batch_size:
            print(f"Warning: Batch size {batch_size} exceeds max {self.max_batch_size}. Truncating.")
            input_data = input_data[:self.max_batch_size]
            batch_size = self.max_batch_size
            
        # 1. Update TRT Context with current batch size
        self.context.set_input_shape("pixel_values", (batch_size, 3, 384, 384))
        
        # 2. Copy data to pinned memory
        np.copyto(self.inputs['pixel_values']['host'][:batch_size], input_data)
        
        # 3. Async Host -> Device
        cuda.memcpy_htod_async(
            self.inputs['pixel_values']['device'], 
            self.inputs['pixel_values']['host'], 
            self.stream
        )
        
        # 4. Execute TRT Graph
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # 5. Async Device -> Host
        cuda.memcpy_dtoh_async(
            self.outputs['image_embedding']['host'], 
            self.outputs['image_embedding']['device'], 
            self.stream
        )
        
        self.stream.synchronize()
        
        # Output shape is [batch_size, 768]. Return as list of numpy arrays
        embeddings = self.outputs['image_embedding']['host'].reshape(self.max_batch_size, -1)
        return [embeddings[i].copy() for i in range(batch_size)]

    def encode_image(self, image_bgr):
        embeddings = self.encode_images_batch([image_bgr])
        return embeddings[0] if embeddings else None

    # --- CROPPING & MATH ---
    def prepare_crops(self, cv_bgr, mask_uint8, bbox_xyxy):
        # (This function remains exactly the same as your original script)
        height, width = cv_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1: return None, None
        
        img_crop = cv_bgr[y1:y2, x1:x2]
        unmasked_crop = img_crop.copy()
        mask_crop = mask_uint8[y1:y2, x1:x2]
        
        if img_crop.size == 0 or mask_crop.size == 0: return None, None
        
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
        image_vec = np.asarray(image_embedding, dtype=np.float64).flatten()
        text_vec = np.asarray(text_embedding, dtype=np.float64).flatten()
        dot_product = float(np.dot(image_vec, text_vec))
        logits = (dot_product * float(self.cached_logit_scale)) + float(self.cached_logit_bias)
        return float(np.float64(logits))

    def compute_match_score(self, image_embedding, text_embedding) -> float:
        logits = self.compute_match_logit(image_embedding, text_embedding)
        return self._safe_sigmoid(logits) * 100.0

    def compute_blended_match_score(self, masked_embedding, unmasked_embedding, text_embedding) -> float:
        masked_score = self.compute_match_score(masked_embedding, text_embedding)
        unmasked_score = self.compute_match_score(unmasked_embedding, text_embedding)
        return (self.masked_score_weight * masked_score) + (self.unmasked_score_weight * unmasked_score)
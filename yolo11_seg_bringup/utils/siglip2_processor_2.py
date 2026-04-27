import cv2
import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from time import perf_counter
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
        self.mean_gpu = torch.tensor([0.5, 0.5, 0.5], device='cuda').view(1, 1, 1, 3)
        self.std_gpu = torch.tensor([0.5, 0.5, 0.5], device='cuda').view(1, 1, 1, 3)

        # --- 2. Load PyTorch Text Encoder ---
        print(f"[SIGLIPProcessor] Loading PyTorch Text Encoder: {self.model_name}")
        self.processor = AutoProcessor.from_pretrained(self.model_name, local_files_only=True)
        self.text_model = AutoModel.from_pretrained(self.model_name, local_files_only=True).to("cuda").eval()
        
        # Cache Logit Scale/Bias
        with torch.no_grad():
            self.cached_logit_scale = float(self.text_model.logit_scale.exp().item()) if hasattr(self.text_model, "logit_scale") else 1.0
            self.cached_logit_bias = float(self.text_model.logit_bias.item()) if hasattr(self.text_model, "logit_bias") else -10.0
        self.last_batch_timing_ms = {}

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
        """PyTorch GPU-Accelerated batch preprocessing."""
        batch_resized = []
        for img in images_bgr:
            if img is None or img.size == 0: continue
            
            # Fast CPU resize (uint8)
            img_resized = cv2.resize(img, (384, 384), interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            batch_resized.append(img_rgb)
            
        if not batch_resized: return None
        
        # 1. Stack as raw bytes (uint8)
        stacked_uint8 = np.stack(batch_resized) # Shape: (B, 384, 384, 3)
        
        # 2. Transfer bytes to GPU immediately (Very fast because uint8 is small)
        tensor_gpu = torch.from_numpy(stacked_uint8).to('cuda', non_blocking=True)
        
        # 3. Vectorized Math ON THE GPU (Zero CPU overhead)
        tensor_gpu = tensor_gpu.float() / 255.0
        tensor_gpu = (tensor_gpu - self.mean_gpu) / self.std_gpu
        
        # 4. Transpose on GPU: (B, H, W, C) -> (B, C, H, W)
        tensor_gpu = tensor_gpu.permute(0, 3, 1, 2).contiguous()
        
        return tensor_gpu # Returns a PyTorch CUDA Tensor

    def encode_images_batch(self, images_bgr: list, return_timing: bool = False):
        """Encodes a batch using PyTorch GPU preprocessing + TensorRT.

        Set return_timing=True to also receive a per-step host-side timing dictionary.
        """
        total_start = perf_counter()
        timing = {
            "preprocess_ms": 0.0,
            "set_shape_ms": 0.0,
            "torch_sync_ms": 0.0,
            "dtod_enqueue_ms": 0.0,
            "inference_enqueue_ms": 0.0,
            "inference_host_call_only_ms": 0.0,
            "dtoh_enqueue_ms": 0.0,
            "stream_sync_ms": 0.0,
            "postprocess_ms": 0.0,
            "total_ms": 0.0,
            "batch_size": 0,
        }

        # 1. Get the preprocessed PyTorch tensor (already on GPU)
        t0 = perf_counter()
        tensor_gpu = self._preprocess_images(images_bgr)
        timing["preprocess_ms"] = (perf_counter() - t0) * 1000.0
        if tensor_gpu is None:
            timing["total_ms"] = (perf_counter() - total_start) * 1000.0
            self.last_batch_timing_ms = timing
            if return_timing:
                return [], timing
            return []

        batch_size = tensor_gpu.size(0)
        if batch_size > self.max_batch_size:
            tensor_gpu = tensor_gpu[:self.max_batch_size]
            batch_size = self.max_batch_size
        timing["batch_size"] = int(batch_size)

        # 2. Update TRT Context shape
        t1 = perf_counter()
        self.context.set_input_shape("pixel_values", (batch_size, 3, 384, 384))
        timing["set_shape_ms"] = (perf_counter() - t1) * 1000.0

        # 3. Synchronize PyTorch to ensure math is finished before TRT starts
        t2 = perf_counter()
        torch.cuda.current_stream().synchronize()
        timing["torch_sync_ms"] = (perf_counter() - t2) * 1000.0

        # 4. Device-to-Device Copy (GPU to GPU, bypassing CPU completely)
        t3 = perf_counter()
        cuda.memcpy_dtod_async(
            self.inputs['pixel_values']['device'],  # Destination TRT Buffer
            tensor_gpu.data_ptr(),                  # Source PyTorch Tensor Pointer
            tensor_gpu.numel() * tensor_gpu.element_size(), # Size in bytes
            self.stream
        )
        timing["dtod_enqueue_ms"] = (perf_counter() - t3) * 1000.0

        # 5. Execute TRT Graph
        t4 = perf_counter()
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        timing["inference_enqueue_ms"] = (perf_counter() - t4) * 1000.0
        timing["inference_host_call_only_ms"] = timing["inference_enqueue_ms"]

        # 6. Async Device -> Host for the tiny output embeddings
        t5 = perf_counter()
        cuda.memcpy_dtoh_async(
            self.outputs['image_embedding']['host'],
            self.outputs['image_embedding']['device'],
            self.stream
        )
        timing["dtoh_enqueue_ms"] = (perf_counter() - t5) * 1000.0

        t6 = perf_counter()
        self.stream.synchronize()
        timing["stream_sync_ms"] = (perf_counter() - t6) * 1000.0

        # Reshape and return
        t7 = perf_counter()
        embeddings = self.outputs['image_embedding']['host'].reshape(self.max_batch_size, -1)
        output_embeddings = [embeddings[i].copy() for i in range(batch_size)]
        timing["postprocess_ms"] = (perf_counter() - t7) * 1000.0
        timing["total_ms"] = (perf_counter() - total_start) * 1000.0
        self.last_batch_timing_ms = timing

        if return_timing:
            return output_embeddings, timing
        return output_embeddings

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
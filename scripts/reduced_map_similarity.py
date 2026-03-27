#!/usr/bin/env python3

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

# Allow running as a standalone script without editable install.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from yolo11_seg_bringup.utils.clip_processor_validator import CLIPProcessorValidator


DEFAULT_REDUCED_MAP = "/workspaces/ros2_ws/src/yolo11_seg_bringup/config/reduced_map.json"
DEFAULT_CLIP_PROMPT = "/workspaces/ros2_ws/src/yolo11_seg_bringup/config/clip_prompt.json"


def load_json(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def build_validator_prompt_ensemble(clip_prompt: str, clip_processor: CLIPProcessorValidator) -> List[str]:
    if not isinstance(clip_prompt, str):
        return []
    clean = clip_prompt.strip()
    if not clean:
        return []
    return list(dict.fromkeys(clip_processor.build_prompt_list(clean)))


def parse_reduced_map_objects(data: Dict) -> List[Tuple[str, Dict]]:
    if not isinstance(data, dict):
        return []

    objects = data.get("objects", {})
    if not isinstance(objects, dict):
        return []

    parsed = []
    for object_id, payload in objects.items():
        if not isinstance(payload, dict):
            continue
        parsed.append((str(object_id), payload))
    return parsed


def compute_sorted_similarities(
    objects: List[Tuple[str, Dict]],
    text_embedding: np.ndarray,
    clip_processor: CLIPProcessorValidator,
) -> List[Dict]:
    results = []

    for object_id, payload in objects:
        embedding = payload.get("embedding", None)
        obj_class = payload.get("class", "")

        if not isinstance(embedding, list) or len(embedding) == 0:
            score = 0.0
        else:
            image_embedding = np.asarray(embedding, dtype=np.float32)
            score = float(clip_processor.compute_match_score(image_embedding, text_embedding))

        results.append(
            {
                "id": object_id,
                "class": str(obj_class),
                "similarity": score,
                "pose": payload.get("pose", {}),
                "occurrences": int(payload.get("occurrences", 0)),
            }
        )

    results.sort(key=lambda item: item["similarity"], reverse=True)
    return results


def print_ranked_results(results: List[Dict], clip_prompt: str) -> None:
    print("\n" + "=" * 90)
    print(f"Similarity ranking for clip_prompt='{clip_prompt}'")
    print("=" * 90)

    if not results:
        print("No objects available in reduced map.")
        return

    for rank, item in enumerate(results, start=1):
        pose = item.get("pose", {})
        x = pose.get("x", None)
        y = pose.get("y", None)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            pose_str = f"({x:.1f}, {y:.1f})"
        else:
            pose_str = "N/A"

        print(
            f"{rank:02d}. id={item['id']} class={item['class']:<10} "
            f"similarity={item['similarity']:.2f} pose={pose_str} occ={item['occurrences']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute CLIP similarity between reduced_map objects and clip_prompt.json "
            "using the same scoring path used in llm_transformers.py"
        )
    )
    parser.add_argument("--reduced-map", default=DEFAULT_REDUCED_MAP, help="Path to reduced_map.json")
    parser.add_argument("--clip-prompt", default=DEFAULT_CLIP_PROMPT, help="Path to clip_prompt.json")
    parser.add_argument("--model-name", default="ViT-B-16-SigLIP", help="open_clip model name")
    parser.add_argument("--pretrained", default="webli", help="open_clip pretrained checkpoint")
    args = parser.parse_args()

    reduced_map_data = load_json(args.reduced_map)
    clip_prompt_data = load_json(args.clip_prompt)

    clip_prompt = clip_prompt_data.get("clip_prompt", "")
    if not isinstance(clip_prompt, str) or not clip_prompt.strip():
        raise ValueError(f"Invalid or empty 'clip_prompt' in {args.clip_prompt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model '{args.model_name}' on {device} ...")
    clip_processor = CLIPProcessorValidator(
        device=device,
        model_name=args.model_name,
        pretrained=args.pretrained,
    )

    prompt_ensemble = build_validator_prompt_ensemble(clip_prompt, clip_processor)
    if not prompt_ensemble:
        raise ValueError("Failed to build validator prompt ensemble from clip_prompt")

    print(f"Encoding {len(prompt_ensemble)} validator-style prompts ...")
    text_embedding = clip_processor.encode_text(prompt_ensemble)
    if text_embedding is None:
        raise RuntimeError("Failed to encode text embedding")

    objects = parse_reduced_map_objects(reduced_map_data)
    ranked = compute_sorted_similarities(objects, text_embedding, clip_processor)
    print_ranked_results(ranked, clip_prompt)


if __name__ == "__main__":
    main()

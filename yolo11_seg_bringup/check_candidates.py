#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import torch
import os
import json

from .utils.clip_processor import CLIPProcessor

from yolo11_seg_interfaces.srv import CheckCandidates

class ClipServiceNode(Node):

    def __init__(self):
        super().__init__('clip_service_node')
        
        self.srv = self.create_service(
            CheckCandidates, 
            'check_candidates', 
            self.check_candidates_callback
        )

        self.declare_parameter('CLIP_model_name', 'ViT-B-16-SigLIP')
        self.declare_parameter('map_file_path', '/home/sensor/ros2_ws/src/yolo11_seg_bringup/config/map.json')

        self.CLIP_model_name = self.get_parameter('CLIP_model_name').value
        self.map_file_path = self.get_parameter('map_file_path').value

        self.text_prompt = 'a photo of a chair'

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Loading CLIP model on device: {self.device}\n")
        self.clip = CLIPProcessor(
            device=self.device, 
            model_name=self.CLIP_model_name, 
            pretrained="webli"
        )

        self.goal_text_embedding = None
        self.current_clip_prompt = None

    def check_candidates_callback(self, request, response):
        """
        Filter map entries by candidate IDs, build an ID->embedding vocabulary,
        and rank candidates by CLIP text-image similarity.
        """
        response.match_found = False
        response.best_candidate_id = ""
        response.max_score = 0.0

        try:
            candidates = list(request.candidate_ids or [])
            clip_prompt = request.prompt

            if not os.path.exists(self.map_file_path):
                self.get_logger().warn(f"Map file not found: {self.map_file_path}")
                return response

            if clip_prompt != self.current_clip_prompt:
                self.current_clip_prompt = clip_prompt
                self.goal_text_embedding = self.clip.encode_text(clip_prompt)
                if self.goal_text_embedding is None:
                    self.get_logger().warn("Prompt is empty; cannot compute text embedding.")
                    return response
                self.get_logger().info(f"Updated goal embedding for prompt: {clip_prompt}")

            with open(self.map_file_path, 'r') as f:
                data = json.load(f)

            vocabulary = {}
            for cid in candidates:
                obj = data.get(cid)
                if not obj:
                    continue
                embedding = obj.get("image_embedding")
                if embedding is None:
                    continue
                vocabulary[cid] = embedding

            if not vocabulary:
                self.get_logger().warn("No candidate embeddings found in map for provided IDs.")
                return response

            best_id = ""
            best_score = float("-inf")
            for cid, embedding in vocabulary.items():
                score = self.clip.compute_sigmoid_probs(embedding, self.goal_text_embedding)
                if score is None:
                    continue
                if score > best_score:
                    best_score = score
                    best_id = cid

            if best_id:
                response.match_found = True
                response.best_candidate_id = best_id
                response.max_score = float(best_score)
                self.get_logger().info(f"Best candidate: {best_id} (score={best_score:.4f})")
            else:
                self.get_logger().warn("Unable to compute similarity scores for candidates.")

        except Exception as e:
            self.get_logger().error(f"Error processing candidates: {e}")

        return response


def main(args=None):
    rclpy.init(args=args)
    node = ClipServiceNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
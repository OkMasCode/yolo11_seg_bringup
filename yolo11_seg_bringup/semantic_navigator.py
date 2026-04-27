#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import math

# Built-in TF2 libraries for finding the robot's position
from tf2_ros import Buffer, TransformListener

# Built-in Nav2 message for sending driving goals
from nav2_msgs.action import NavigateToPose

# Your custom message interface
from yolo11_seg_interfaces.msg import SimilarityCentroidArray


class SemanticNavigator(Node):
    def __init__(self):
        super().__init__('semantic_navigator')

        # 1. Setup TF2 to listen to the robot's location
        # A buffer stores the history of transforms, and the listener fills that buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 2. Setup the Action Client to communicate with Nav2
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # 3. Setup the Subscriber to listen to your C++ SemanticMapper
        self.subscription = self.create_subscription(
            SimilarityCentroidArray,
            '/similarity_centroids_data',
            self.cluster_callback,
            10
        )

        # 4. Tuning Weights for the Cost Function
        # Increase weight_distance to make the robot prefer nearby areas
        # Increase weight_similarity to make the robot prefer high-similarity targets
        self.weight_distance = 1.0
        self.weight_similarity = 5.0  

        # State flag to ensure we don't send new goals while the robot is already driving
        self.is_navigating = False

        self.get_logger().info("[INIT] Semantic Navigator Node initialized")
        self.get_logger().info("[INIT] TF2 buffer created")
        self.get_logger().info("[INIT] Nav2 action client created for 'navigate_to_pose'")
        self.get_logger().info("[INIT] Subscribed to '/similarity_centroids_data'")
        self.get_logger().info("[INIT] Waiting for clusters...")

    def cluster_callback(self, msg: SimilarityCentroidArray):
        """
        This function triggers every time the C++ node publishes new clusters.
        """
        self.get_logger().info(f"[CALLBACK] Received {len(msg.clusters)} clusters")
        
        # If the robot is already driving to a goal, or there are no clusters, do nothing
        if self.is_navigating:
            self.get_logger().debug("[CALLBACK] Robot already navigating, skipping")
            return
        
        if not msg.clusters:
            self.get_logger().debug("[CALLBACK] No clusters in message")
            return

        # Look up the robot's current position on the map
        self.get_logger().info("[CALLBACK] Looking up robot position from TF2...")
        try:
            # We want the transform from the global 'map' to the robot's 'base_link'
            transform = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            robot_x = transform.transform.translation.x
            robot_y = transform.transform.translation.y
            self.get_logger().info(f"[CALLBACK] Robot position: ({robot_x:.2f}, {robot_y:.2f})")
        except Exception as e:
            self.get_logger().error(f"[CALLBACK] TF2 lookup failed: {e}")
            return

        best_cluster = None
        lowest_cost = float('inf')
        best_distance = 0.0

        self.get_logger().info("[CALLBACK] Evaluating all clusters...")
        # Evaluate every cluster to find the most optimal one
        for idx, cluster in enumerate(msg.clusters):
            target_x = cluster.position.x
            target_y = cluster.position.y
            similarity = cluster.similarity 

            # Calculate the straight-line distance using the built-in math library
            distance = math.hypot(target_x - robot_x, target_y - robot_y)

            # Calculate the cost: lower distance and higher similarity result in a lower cost
            cost = (distance * self.weight_distance) + (similarity * self.weight_similarity)
            
            self.get_logger().debug(f"[CALLBACK]   Cluster {idx}: pos=({target_x:.2f}, {target_y:.2f}), sim={similarity:.3f}, dist={distance:.2f}, cost={cost:.2f}")

            if cost < lowest_cost:
                lowest_cost = cost
                best_cluster = cluster
                best_distance = distance
                self.get_logger().debug(f"[CALLBACK]     -> New best cluster!")

        # If we successfully found a cluster, send the goal to Nav2
        if best_cluster:
            self.get_logger().info(
                f"[CALLBACK] *** TARGET SELECTED *** Distance: {best_distance:.2f}m, "
                f"Similarity: {best_cluster.similarity:.2f}, Cost: {lowest_cost:.2f}"
            )
            self.get_logger().info(f"[CALLBACK] Sending navigation goal...")
            self.send_nav_goal(best_cluster.position)
        else:
            self.get_logger().warn("[CALLBACK] No valid cluster found")


    def send_nav_goal(self, target_position):
        """
        Constructs the action goal and sends it asynchronously to Nav2.
        """
        self.get_logger().info('[SEND_GOAL] Waiting for Nav2 action server to become available...')
        try:
            self.nav_client.wait_for_server(timeout_sec=5.0)
            self.get_logger().info('[SEND_GOAL] Nav2 action server is available')
        except Exception as e:
            self.get_logger().error(f'[SEND_GOAL] Nav2 action server not available: {e}')
            self.is_navigating = False
            return

        # Create the standard NavigateToPose goal message
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        
        # Extract the X and Y coordinates from your custom message point
        goal_msg.pose.pose.position.x = target_position.x
        goal_msg.pose.pose.position.y = target_position.y
        goal_msg.pose.pose.position.z = 0.0 
        
        # Set a neutral forward-facing orientation (quaternion)
        goal_msg.pose.pose.orientation.w = 1.0
        goal_msg.pose.pose.orientation.x = 0.0
        goal_msg.pose.pose.orientation.y = 0.0
        goal_msg.pose.pose.orientation.z = 0.0

        self.get_logger().info(f'[SEND_GOAL] Goal constructed: target=({target_position.x:.2f}, {target_position.y:.2f})')

        # Lock the state so we don't spam the server
        self.is_navigating = True
        self.get_logger().info('[SEND_GOAL] Locked navigation state')
        
        # Send the goal asynchronously
        self.get_logger().info('[SEND_GOAL] Sending goal async to Nav2...')
        self._send_goal_future = self.nav_client.send_goal_async(goal_msg)
        
        # Attach a callback to run when the server responds to our request
        self._send_goal_future.add_done_callback(self.goal_response_callback)
        self.get_logger().info('[SEND_GOAL] Goal callback attached, waiting for response...')


    def goal_response_callback(self, future):
        """
        Checks if Nav2 accepted or rejected the goal we just sent.
        """
        self.get_logger().info('[GOAL_RESP] Nav2 goal response received')
        try:
            goal_handle = future.result()
        except Exception as e:
            self.get_logger().error(f'[GOAL_RESP] Exception getting goal handle: {e}')
            self.is_navigating = False
            return
        
        if not goal_handle.accepted:
            self.get_logger().error('[GOAL_RESP] Nav2 rejected the goal')
            self.is_navigating = False
            return

        self.get_logger().info('[GOAL_RESP] Nav2 accepted the goal! Robot driving now...')
        
        # Wait asynchronously for the robot to actually arrive at the destination
        self.get_logger().info('[GOAL_RESP] Waiting for navigation result...')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)


    def get_result_callback(self, future):
        """
        Executes when the robot has finished driving (arrived, failed, or cancelled).
        """
        self.get_logger().info('[RESULT] Navigation result received')
        try:
            result = future.result()
            self.get_logger().info(f'[RESULT] Nav2 result status: {result.result}')
        except Exception as e:
            self.get_logger().error(f'[RESULT] Exception getting result: {e}')
        
        # Unlock the state so the cluster_callback can select the next target
        self.is_navigating = False 
        self.get_logger().info('[RESULT] Unlocked navigation state, ready for next target...')


def main(args=None):
    # Initialize the ROS 2 Python client library
    rclpy.init(args=args)
    
    # Create the node
    node = SemanticNavigator()
    
    # Keep the node alive and listening for messages
    rclpy.spin(node)
    
    # Clean up when the node is shut down (Ctrl+C)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
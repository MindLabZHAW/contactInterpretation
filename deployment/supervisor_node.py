#!/usr/bin/env python3
"""
Supervisor Node (The "Brain") - Updated for Dynamic Model Loading

- On startup, prompts the user to select a model directory.
- Scans the selected directory for models and prompts the user to select one.
- Parses the model filename to configure hyperparameters (layers, hidden size, sequence length).
- Instantiates the correct model architecture (Detection or Localization).
- Subscribes to the robot's state and runs the loaded model for inference.
- Publishes a "STOP" command on /robot_command if contact is detected.
- Monitors and reports the connection status to the robot.
# How to run?

#### 1st  Step: unlock robot
	-turn on the robot (wait until it has a solid yellow)
	-connect to the robot desk with the ID (172.16.0.2 or 192.168.15.33)
	-unlock the robot
	-the robot light should be blue
	-unlock the robot and activate FCI

#### 2nd Step: run frankapy

open an terminal

	conda activate frankapyenv
	bash robotAPI/frankapy/bash_scripts/start_control_pc.sh -i localhost


#### 3rd Step: run robot node

open another terminal 
	conda activate frankapyenv
	source /opt/ros/noetic/setup.bash
	source robotAPI/franka-interface/catkin_ws/devel/setup.bash --extend
	source robotAPI/frankapy/catkin_ws/devel/setup.bash --extend
"""
import rospy
import numpy as np
import os
import re
import torch
import sys
from std_msgs.msg import String
from franka_interface_msgs.msg import RobotState
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg

# --- Dynamically add the project root to the Python path ---
# This allows us to import from the 'pipelines' directory.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from  pipelines.models.cnnLSTM_contactDetection import cnnLSTM as ContactDetectionModel
from  pipelines.models.cnnLSTM_contactLocalization import cnnLSTM as ContactLocalizationModel


class SupervisorNode:
    def __init__(self):
        rospy.init_node('supervisor_node')
        
        # This constant defines the expected degrees of freedom from this robot
        self.robot_dof = 7

        # --- Dynamic Model Selection and Loading ---
        self.model, self.window_length = self._select_and_load_model()
        if self.model is None:
            rospy.logerr("No model was loaded. Shutting down.")
            rospy.signal_shutdown("Model loading failed.")
            return

        # --- ROS Comms Setup ---
        self.command_pub = rospy.Publisher('/robot_command', String, queue_size=10)
        self.model_output_pub = rospy.Publisher('/model_output', numpy_msg(Floats), queue_size=10)
        
        # --- Detection State ---
        # The window should store data as (DoF, Window Length)
        self.window = np.zeros([self.robot_dof, self.window_length])
        self.contact_detected_latch = False

        # --- Connection Monitoring ---
        self.is_connected = False
        
        # Subscriber is created first so we can check its connection status
        self.robot_state_sub = rospy.Subscriber(
            "/robot_state_publisher_node_1/robot_state",
            RobotState,
            self.robot_state_callback
        )

        # Timer checks the subscriber's connection status periodically.
        self.connection_check_timer = rospy.Timer(rospy.Duration(2.0), self.check_connection)

        rospy.loginfo("Supervisor node initialized. Waiting for connection to robot...")

    def check_connection(self, event):
        """
        Timer callback that directly checks the number of publishers
        connected to our robot state subscriber.
        """
        if self.robot_state_sub.get_num_connections() > 0:
            if not self.is_connected:
                rospy.loginfo("Connection to robot state publisher established.")
                self.is_connected = True
        else:
            if self.is_connected:
                rospy.logwarn("Connection to robot state publisher lost!")
                self.is_connected = False
            else:
                # This message will repeat until a connection is made
                rospy.logwarn_throttle(5, "Waiting for connection to robot state publisher...")

    def _parse_model_name(self, filename):
        """
        Parses hyperparameters from a model's filename.
        It's now robust to missing 'numLayer'.
        """
        params = {}
        try:
            hidden_size_match = re.search(r'hiddenSize(\d+)', filename)
            seq_num_match = re.search(r'seq_num(\d+)', filename)

            if not all([hidden_size_match, seq_num_match]):
                rospy.logwarn(f"Filename '{filename}' is missing 'hiddenSize' or 'seq_num'.")
                return None
            
            # Use a default for num_layers if not found, to match model's default
            num_layers_match = re.search(r'numLayer(\d+)', filename)
            params['num_layers'] = int(num_layers_match.group(1)) if num_layers_match else 3
            params['hidden_size'] = int(hidden_size_match.group(1))
            params['seq_num'] = int(seq_num_match.group(1))
            return params
        except (AttributeError, ValueError) as e:
            rospy.logwarn(f"Could not parse filename '{filename}': {e}.")
            return None

    def _select_and_load_model(self):
        """
        Scans for models in user-selected subdirectory, infers architecture,
        and loads the selected model.
        """
        base_models_dir = os.path.join(os.getcwd(), 'pipelines', 'trained_models')
        if not os.path.isdir(base_models_dir):
            rospy.logerr(f"Base model directory not found at: {base_models_dir}")
            return None, -1

        # --- Step 1: Let the user select a subdirectory ---
        try:
            subdirectories = [d for d in os.listdir(base_models_dir) if os.path.isdir(os.path.join(base_models_dir, d))]
            if not subdirectories:
                rospy.logerr(f"No subdirectories found in {base_models_dir}.")
                return None, -1
            
            print("\n--- Please select a model directory ---")
            for i, dirname in enumerate(subdirectories):
                print(f"[{i}] {dirname}")
            
            dir_choice = int(input("Please select a directory by number: "))
            selected_dir_name = subdirectories[dir_choice]
            search_dir = os.path.join(base_models_dir, selected_dir_name)

        except (ValueError, IndexError):
            rospy.logerr("Invalid directory selection.")
            return None, -1

        # --- Step 2: Find and select a model within the chosen directory ---
        available_models = []
        for root, _, files in os.walk(search_dir):
            for file in files:
                if file.endswith('.pth'):
                    full_path = os.path.join(root, file)
                    display_path = os.path.relpath(full_path, search_dir)
                    available_models.append((display_path, full_path))
        
        if not available_models:
            rospy.logerr(f"No '.pth' models found in the selected directory: {selected_dir_name}")
            return None, -1

        print(f"\n--- Available Models in '{selected_dir_name}' ---")
        for i, (display_path, _) in enumerate(available_models):
            print(f"[{i}] {display_path}")
        
        try:
            model_choice = int(input("Please select a model by number: "))
            display_path, selected_full_path = available_models[model_choice]
            selected_filename = os.path.basename(selected_full_path)
        except (ValueError, IndexError):
            rospy.logerr("Invalid model selection.")
            return None, -1
        
        params = self._parse_model_name(selected_filename)
        if params is None:
            return None, -1

        # --- Step 3: Instantiate and load the model ---
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(selected_full_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        final_window_length = params['seq_num']

        model_type = 'contactDetection' if 'contactDetection' in display_path or 'contact_detection' in display_path else \
                     'contactLocalization' if 'contactLocalization' in display_path or 'contact_localization' in display_path else None
        
        rospy.loginfo(f"Loading model '{selected_filename}' with specified params:")
        rospy.loginfo(f"  - Type: {model_type}, Window: {final_window_length}, Bidirectional: True")
        rospy.loginfo(f"  - Filename params: {params}")

        ModelClass = ContactDetectionModel if model_type == 'contactDetection' else ContactLocalizationModel
        
        # Per user request, instantiate the model with seq_num as num_features_joints
        # and assume bidirectional is always true.
        model = ModelClass(
            num_features_joints=final_window_length,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            bidirectional=True
        )

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        self.device = device
        
        rospy.loginfo(f"Successfully loaded model: {selected_filename}")
        return model, final_window_length

    def robot_state_callback(self, data):
        """Runs inference only when connected and if contact hasn't been detected."""
        if not self.is_connected or self.contact_detected_latch:
            return

        # --- Data Preparation: Only use joint position error (e_q) ---
        e_q = np.array(data.q_d) - np.array(data.q)

        # Reshape the new data to a column vector (DoF, 1)
        new_column = e_q.reshape((self.robot_dof, 1))
        
        # Append the new column of features to the window, and drop the oldest column
        self.window = np.append(self.window[:, 1:], new_column, axis=1)
        
        # --- Model Inference ---
        # Reshape window to (batch_size, num_joints, window_length) for the model, as per paper
        data_input = torch.tensor([self.window], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            contact_result = self.model.prediction(data_input).item()

        # --- Action based on Inference ---
        if contact_result == 1:
            rospy.logwarn("CONTACT DETECTED! Sending STOP command to motion node.")
            self.command_pub.publish("STOP")
            self.contact_detected_latch = True

    def run(self):
        """Keeps the node alive."""
        rospy.spin()

if __name__ == '__main__':
    try:
        supervisor = SupervisorNode()
        if supervisor.model:
            supervisor.run()
    except rospy.ROSInterruptException:
        pass


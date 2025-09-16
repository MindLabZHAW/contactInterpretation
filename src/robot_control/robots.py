import numpy as np
import time
import logging
import re
import threading
from robot_control.robot_interface import RobotInterface
from typing import List, Dict, Callable, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import frankx
except ImportError:
    frankx = None


class SimulationRobot(RobotInterface):
    # ... (This class is correct and remains unchanged) ...
    def __init__(self, csv_file_path: str, selected_features: List[str]):

        if pd is None:
            raise ImportError("The 'pandas' library is not installed.")

        self.file_path = csv_file_path
        self.selected_features = selected_features
        self.data = None
        self.current_step_index = 0
        self.total_steps = 0
        self._is_performing_action = False
        self.ground_truth_link = 0

        match = re.search(r'link(\d+)', self.file_path)
        if match:
            self.ground_truth_link = int(match.group(1))
            logging.info(f"✅ Ground truth contact link parsed from file path: {self.ground_truth_link}")
        else:
            logging.warning(f"Could not parse link number from file path: {self.file_path}. Localization ground truth will be 0.")

        print(f"🤖 SimulationRobot initialized for data file: {self.file_path}")

    def connect(self) -> bool:
        print("Connecting to simulation...")
        try:
            full_data = pd.read_csv(self.file_path)

            start_index = 0#len(full_data) // 2
            self.data = full_data.iloc[start_index:].reset_index(drop=True)

            self.total_steps = len(self.data)
            self.current_step_index = 0

            required_cols = self.selected_features + ['label']
            if not all(col in self.data.columns for col in required_cols):
                logging.error(f"One or more required columns are missing in {self.file_path}.")
                return False
            print(f"✅ Simulation data loaded successfully with {self.total_steps} timesteps (processing second half).")
            return True
        except Exception as e:
            logging.error(f"Failed to load or parse simulation CSV: {e}")
            return False

    def disconnect(self) -> None:
        print("\nDisconnected from simulation.")

    def get_data(self) -> dict:
        if self.current_step_index >= self.total_steps:
            self.current_step_index = self.total_steps - 1

        row = self.data.iloc[self.current_step_index]
        feature_vector = row[self.selected_features].values
        detection_label = int(row['label'])

        localization_label = self.ground_truth_link if detection_label == 1 else 0

        self.current_step_index += 1
        #features_dict = {f'e{i}': val for i, val in enumerate(feature_vector)}
        features_dict = {name: val for name, val in zip(self.selected_features, feature_vector)}

        return {
            "features": features_dict,
            "label": detection_label,
            "contact_link": localization_label
        }

    def send_action(self, action_data: dict):
        self._is_performing_action = True

    def is_performing_action(self) -> bool:
        if self.current_step_index >= self.total_steps:
            self._is_performing_action = False
        return self._is_performing_action

    def stop(self) -> None:
        self._is_performing_action = False


class FrankaRobot(RobotInterface):
    """
    Final robust implementation based on the user's suggested architecture.
    Uses a dedicated state-reading thread and a dedicated motion-monitoring thread.
    """
    def __init__(self, ip_address: str, selected_features: List[str]):
        if frankx is None:
            raise ImportError("The 'frankx' library is not installed.")
        
        self.ip_address = ip_address
        self.selected_features = selected_features
        self.robot = None
        self.motion_thread = None # <-- THIS LINE FIXES THE ERROR
        
        # Thread-safe state handling
        self.latest_state = None
        self.state_lock = threading.Lock()
        self.state_reader_thread = None
        self.run_state_reader = False
        self.robot_lock = threading.Lock()
        
        logging.info(f"🤖 FrankaRobot initialized for IP address: {self.ip_address}")

    def _state_reader_loop(self):
        """A background thread that ONLY reads robot state."""
        self.run_state_reader = True
        while self.run_state_reader:
            try:
                with self.robot_lock:
                    state = self.robot.read_once()
                with self.state_lock:
                    self.latest_state = state
            except Exception as e:
                if self.run_state_reader:
                    logging.debug(f"State reader thread info: {e}")
            time.sleep(0.001) # Read at 1000 Hz

    def connect(self) -> bool:
        logging.info(f"Connecting to Franka robot at {self.ip_address}...")
        try:
            self.robot = frankx.Robot(self.ip_address)
            self.robot.set_default_behavior()
            self.robot.recover_from_errors()
            self.robot.set_dynamic_rel(0.05)
            logging.info("Robot dynamics set to 5%.")
            
            # Start the state reader thread
            self.state_reader_thread = threading.Thread(target=self._state_reader_loop)
            self.state_reader_thread.daemon = True
            self.state_reader_thread.start()
            
            return True
        except Exception as e:
            logging.error(f"Failed to connect to Franka robot: {e}")
            return False

    def disconnect(self) -> None:
        self.run_state_reader = False
        if self.state_reader_thread:
            self.state_reader_thread.join()
        if self.robot:
            logging.info("Disconnected from Franka robot.")
        self.robot = None

    def get_data(self) -> dict:
        """Gets the latest state and dynamically builds the feature vector."""
        try:
            with self.state_lock:
                state = self.latest_state
            features_dict = {}

            # --- DYNAMIC FEATURE CALCULATION ---
            for feature in self.selected_features:
                feature_base = re.sub(r'\d+$', '', feature)
                index = int(re.search(r'(\d+)$', feature).group(1)) if re.search(r'(\d+)$', feature) else None

                if feature_base == 'e':
                    joint_error = np.array(state.q_d) - np.array(state.q)
                    if index is not None and index < len(joint_error):
                        features_dict[feature] = joint_error[index]

                # Example for adding other features:
                # if feature_base == 'dq':
                #     if index is not None and index < len(state.dq):
                #         features_dict[feature] = state.dq[index]
            
            return {"features": features_dict, "label": 0, "contact_link": 0}
        except state is None:
            return {}       
    def _execute_and_wait(self, motion: 'frankx.Motion', target_joints: List[float]):
        """
        Starts an async move and then waits for the robot to reach the target pose.
        """
        try:
            with self.robot_lock:
                self.robot.move_async(motion)

            while True:
                with self.state_lock:
                    current_q = self.latest_state.q if self.latest_state else None
                
                if current_q:
                    if np.allclose(current_q, target_joints, atol=1e-3):
                        logging.info("Target pose reached.")
                        break
                time.sleep(0.001)
        except Exception as e:
            logging.error(f"Error during motion execution: {e}")

    def send_action(self, action_data: dict) -> None:
        """Creates and starts the motion-monitoring thread."""
        if not self.robot:
            return

        command = action_data.get("command")
        if command == "move":
            target_joints = action_data.get("target")
            if isinstance(target_joints, list) and len(target_joints) == 7:
                motion = frankx.JointMotion(target_joints)
                self.motion_thread = threading.Thread(target=self._execute_and_wait, args=(motion, target_joints))
                self.motion_thread.start()
            else:
                logging.warning(f"Invalid 'target' for move command.")
        else:
            logging.warning(f"Command '{command}' is not yet implemented.")

    def is_performing_action(self) -> bool:
        """Checks if the motion-monitoring thread is still alive."""
        if self.motion_thread:
            return self.motion_thread.is_alive()
        return False

    def stop(self) -> None:
        if self.robot:
            logging.info("🛑 Halting robot motion.")
            self.robot.stop_motion()
import numpy as np
import time
import logging
import re
import threading
from robot_control.robot_interface import RobotInterface
from typing import List, Dict, Callable, Optional
from threading import Thread
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import frankx
except ImportError:
    frankx = None

from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_control import RTDEControlInterface as RTDEControl

class SimulationRobot(RobotInterface):
    # ... (This class is correct and remains unchanged) ...
    def __init__(self, csv_file_path: str, selected_features: List[str], robot_name: str = "SimulationRobot"):

        if pd is None:
            raise ImportError("The 'pandas' library is not installed.")

        self.file_path = csv_file_path
        self.selected_features = selected_features
        self.data = None
        self.current_step_index = 0
        self.total_steps = 0
        self._is_performing_action = False
        self.ground_truth_link = 0
        self.name = robot_name


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
    def __init__(self, ip_address: str, selected_features: List[str], robot_name: str = "FrankaRobot",set_dynamic_rel: float = 0.05):
        if frankx is None:
            raise ImportError("The 'frankx' library is not installed.")
        
        self.ip_address = ip_address
        self.selected_features = selected_features
        self.robot = None
        self.motion_thread = None
        self.name = robot_name
        self.set_dynamic_rel = set_dynamic_rel
        
        # Thread-safe state handling
        self.latest_state = None
        self.state_lock = threading.Lock()
        self.state_reader_thread = None
        self.run_state_reader = False
        self.robot_lock = threading.Lock()

        # Motion execution flag
        self._stop_motion = False
        
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
            self.robot.set_dynamic_rel(self.set_dynamic_rel)
            logging.info("Robot dynamics set to 5%.")
            
            # Start the state reader thread
            self.state_reader_thread = Thread(target=self._state_reader_loop)
            self.state_reader_thread.daemon = True
            self.state_reader_thread.start()
            
            return True
        except Exception as e:
            logging.error(f"Failed to connect to Franka robot: {e}")
            return False

    def disconnect(self) -> None:
        # Signal state reader to stop and wait for it
        self.run_state_reader = False
        if self.state_reader_thread and self.state_reader_thread.is_alive():
            self.state_reader_thread.join()

        # Signal motion thread to stop and wait for it
        if self.motion_thread and self.motion_thread.is_alive():
            self._stop_motion = True
            self.motion_thread.join()

        if self.robot:
            self.stop()
            logging.info("Disconnected from Franka robot.")
        self.robot = None

    def get_data(self) -> dict:
        """Gets the latest state and dynamically builds the feature vector."""
        with self.state_lock:
            state = self.latest_state if self.latest_state else None
        
        if state is None:
            return {}

        features_dict = {}
        for feature in self.selected_features:
            feature_base = re.sub(r'\d+$', '', feature)
            match = re.search(r'(\d+)$', feature)
            index = int(match.group(1)) if match else None

            if feature_base == 'e' and index is not None:
                joint_error = np.array(state.q_d) - np.array(state.q)
                if index < len(joint_error):
                    features_dict[feature] = joint_error[index]
        
        return {"features": features_dict, "label": 0, "contact_link": 0}

    def _move(self, motion: 'frankx.Motion', target_joints: List[float]):
        """
        Starts an async move and then waits for the robot to reach the target pose.
        """
        try:
            with self.robot_lock:
                # Use move_async to not block the main thread
                self.robot.move_async(motion)

            while not self._stop_motion:
                with self.state_lock:
                    current_q = self.latest_state.q if self.latest_state else None
                
                if current_q and np.allclose(current_q, target_joints, atol=1e-3):
                    logging.info("Target pose reached.")
                    break
                time.sleep(0.01)
        except Exception as e:
            if not self._stop_motion:
                logging.error(f"Error during motion execution: {e}")

    def _wait(self, duration: float):
        """
        Waits for a specified duration, checking for stop signal.
        """
        start_time = time.time()
        logging.info(f"Starting wait for {duration} seconds.")
        while time.time() - start_time < duration and not self._stop_motion:
            time.sleep(0.01)
        
        if not self._stop_motion:
            logging.info("Wait completed.")

    def send_action(self, action_data: dict) -> None:
        """Creates and starts the motion-monitoring thread."""
        if self.is_performing_action():
            logging.warning("Motion thread is already running.")
            return

        self._stop_motion = False
        command = action_data.get("command")

        if command == "move":
            target_joints = action_data.get("joints_positions")
            if isinstance(target_joints, list) and len(target_joints) == 7:
                motion = frankx.JointMotion(target_joints)
                self.motion_thread = Thread(target=self._move, args=(motion, target_joints))
                self.motion_thread.start()
            else:
                logging.warning(f"Invalid 'target' for move command.")
        elif command == "wait":
            duration = action_data.get("duration")
            if isinstance(duration, (int, float)) and duration > 0:
                self.motion_thread = Thread(target=self._wait, args=(duration,))
                self.motion_thread.start()
            else:
                logging.warning(f"Invalid 'duration' for wait command: {duration}")
        else:
            logging.warning(f"Command '{command}' is not yet implemented.")

    def is_performing_action(self) -> bool:
        """Checks if the motion-monitoring thread is still alive."""
        return self.motion_thread is not None and self.motion_thread.is_alive()

    def stop(self) -> None:
        """Stops any ongoing robot motion and signals threads to exit."""
        self._stop_motion = True

        # Signal motion thread to stop and wait for it
        if self.motion_thread and self.motion_thread.is_alive():
            self._stop_motion = True
            self.motion_thread.join()

        if self.robot:
            logging.info("🛑 Halting robot motion.")
            self.robot.stop()

class URRobot(RobotInterface):
    """
    A class to interface with a Universal Robot (UR), following the multi-threaded
    architecture of the FrankaRobot class for non-blocking state reading.
    """
    def __init__(self, ip_address: str, frequency: int = 200, selected_features: Optional[List[str]] = None, robot_name: str = "URRobot"):
        self.ip_address = ip_address
        self.frequency = frequency
        self.selected_features = selected_features if selected_features is not None else []
        self.name = robot_name
        
        self.robot_control: Optional[RTDEControl] = None
        self.robot_receive: Optional[RTDEReceive] = None
        
        # Thread-safe state handling
        self.latest_state: Optional[Dict] = None
        self.state_lock = threading.Lock()
        self.robot_lock = threading.Lock()
        self.state_reader_thread: Optional[Thread] = None
        self.run_state_reader = False
        
        # Motion execution thread
        self._action_thread: Optional[Thread] = None
        self._stop_action = False

        # Constants for torque calculation
        self.k_gains = np.array([1.35, 1.361, 1.355, 0.957, 0.865, 0.893])
        logging.info(f"🤖 URRobot initialized for IP address: {self.ip_address}")

    def _state_reader_loop(self):
        """A background thread that continuously reads the robot's state."""
        self.run_state_reader = True
        logging.info("URRobot state reader thread started.")
        while self.run_state_reader:
            try:
                # --- THIS IS THE FIX ---
                # Collect all required raw data from the robot in one go.
                state_data = {
                    "q_actual": np.array(self.robot_receive.getActualQ()),
                    "q_target": np.array(self.robot_receive.getTargetQ())


                }
                ''''                    
                "i_actual": np.array(self.robot_receive.getActualCurrent()),
                    "dq_actual": np.array(self.robot_receive.getActualQd()),
                    "q_target": np.array(self.robot_receive.getTargetQ()),
                    "dq_target": np.array(self.robot_receive.getTargetQd()),
                    "i_target": np.array(self.robot_receive.getTargetCurrent()),
                    "tau_J_target": np.array(self.robot_receive.getTargetMoment())

                state_data["tau_ext"] = (state_data["i_target"] - state_data["i_actual"]) * self.k_gains
                state_data["de"] = state_data["dq_target"] - state_data["dq_actual"]'''
                
                # --- Pre-calculate all derived features ---
                state_data["e"] = state_data["q_target"] - state_data["q_actual"]
                # ----------------------

                with self.state_lock:
                    self.latest_state = state_data

            except Exception as e:
                if self.run_state_reader:
                    logging.debug(f"URRobot state reader error: {e}")
            
            time.sleep(1.0 / self.frequency)
        logging.info("URRobot state reader thread stopped.")

    def connect(self) -> bool:
        if self.run_state_reader:
            logging.info("UR Robot is already connected.")
            return True
        
        try:
            logging.info(f"Connecting to UR Robot at {self.ip_address}...")
            self.robot_control = RTDEControl(self.ip_address)
            self.robot_receive = RTDEReceive(self.ip_address, self.frequency)
            
            if self.robot_control.isConnected() and self.robot_receive.isConnected():
                # Start the state reader thread
                self.state_reader_thread = Thread(target=self._state_reader_loop)
                self.state_reader_thread.daemon = True
                self.state_reader_thread.start()
                logging.info("--- ✅ UR Robot connected successfully. ---")
                return True
            else:
                logging.error("Failed to establish connection with UR Robot components.")
                return False
        except Exception as e:
            logging.error(f"An error occurred during connection: {e}")
            return False

    def disconnect(self):
        self.run_state_reader = False
        if self.state_reader_thread:
            self.state_reader_thread.join(timeout=1)

        if self._action_thread and self._action_thread.is_alive():
            self._stop_action = True
            self._action_thread.join(timeout=1)

        if self.robot_control and self.robot_control.isConnected():
            try:
                self.stop()
                self.robot_control.stopScript()
            except Exception as e:
                logging.warning(f"Could not stop script cleanly: {e}")
        
        self.robot_control = None
        self.robot_receive = None
        logging.info("--- UR Robot disconnected. ---")

    def get_data(self) -> Optional[Dict]:
        """
        Gets the latest state from the reader thread and builds the feature vector.
        """
        with self.state_lock:
            state = self.latest_state
        
        if state is None:
            return None

        features_dict = {}
        for feature in self.selected_features:
            # Assumes features are named like 'tau_ext_0', 'q_error_1', etc.
            match = re.match(r'([a-zA-Z_]+)(\d+)', feature)
            if match:
                feature_base, index_str = match.groups()
                index = int(index_str)
                
                if feature_base in state and index < len(state[feature_base]):
                    features_dict[feature] = state[feature_base][index]
            elif feature in state: # For features that are not arrays
                 features_dict[feature] = state[feature]
    
        return {"features": features_dict, "label": 0, "contact_link": 0}

    def send_action(self, action_data: Dict):
        """
        Sends a command to the robot. Handles blocking calls by using a background thread.
        """
        if not self.run_state_reader or not self.robot_control:
            logging.error("Cannot send action: Robot is not connected.")
            return

        command = action_data.get("command")
        
        if command == "move":
            # move is blocking, so we run it in a thread to allow monitoring.
            target_joint = np.array(action_data["joints_positions"])
            speed = action_data.get("speed", 0.2)
            accel = action_data.get("acceleration", 0.1)
            
            self._stop_action = False
            self._action_thread = Thread(
                target=self._execute_move, 
                args=(self.robot_control.moveL_FK, target_joint, speed, accel, True)
            )
            self._action_thread.start()

        elif command == "wait":
            duration = action_data.get("duration", 1.0)
            self._stop_action = False
            self._action_thread = Thread(target=self._execute_wait, args=(duration,))
            self._action_thread.start()
            
        else:
            logging.warning(f"Unknown command '{command}' received. Ignoring.")

    def _execute_move(self, move_function, *args):
        """
        Starts an asynchronous move and then enters a loop to monitor
        the robot's state until the target is reached.
        """
        try:
            # Start the move asynchronously
            with self.robot_lock:
                move_function(*args) 
            target = args[0]  
            
            while not self._stop_action:
                with self.state_lock:
                    state = self.latest_state
                
                if state:
                    is_at_target = False
                    current_q = state.get("q_actual")
                    if current_q is not None:
                        is_at_target = np.allclose(current_q, target, atol=1e-3)

                    if is_at_target:
                        logging.info(f"Target reached for target joint.")
                        break # Exit the monitoring loop
                
                time.sleep(0.01) # Check for completion at 100 Hz
            #self.robot_control.stopL(5.0)  # Ensure the robot is stopped after the move
        except Exception as e:
            logging.error(f"An error occurred during robot movement: {e}")

    def _execute_wait(self, duration: float):
        """Target function for the action thread to wait."""
        start_time = time.time()
        while time.time() - start_time < duration and not self._stop_action:
            time.sleep(0.01)

    def is_performing_action(self) -> bool:
        """
        Checks if the robot is currently executing a threaded action.
        """
        return self._action_thread is not None and self._action_thread.is_alive()    
    
    def stop(self) -> None:
        """Stops any ongoing robot motion."""
        self._stop_action = True  # Signal waiting threads to stop
        
        if self._action_thread and self._action_thread.is_alive():
            self._stop_action = True
            self._action_thread.join(timeout=1)

        if self.robot_control and self.robot_control.isConnected():
            try:
                logging.info("🛑 Halting UR robot motion.")
                # Use stopL with a high acceleration to stop linear moves quickly.
                with self.robot_lock:
                    self.robot_control.stopL(5.0)
            except Exception as e:
                logging.error(f"Failed to send stop command to UR robot: {e}")
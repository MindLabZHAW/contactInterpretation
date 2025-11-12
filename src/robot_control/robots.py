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

# Import the new, maintained library
try:
    # The pip package is 'franky-control', but the module is 'franky'
    import franky
except ImportError:
    franky = None

from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_control import RTDEControlInterface as RTDEControl

class SimulationRobot(RobotInterface):
    """
    Simulated robot that reads from a CSV file.
    It dynamically builds its feature vector based on 'selected_features'.
    """
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
            
            required_cols = self.selected_features + ['label']
            if not all(col in full_data.columns for col in required_cols):
                missing = [col for col in required_cols if col not in full_data.columns]
                logging.error(f"One or more required columns are missing in {self.file_path}. Missing: {missing}")
                return False

            start_index = 0
            self.data = full_data.iloc[start_index:].reset_index(drop=True)
            self.total_steps = len(self.data)
            self.current_step_index = 0

            print(f"✅ Simulation data loaded successfully with {self.total_steps} timesteps.")
            return True
        except Exception as e:
            logging.error(f"Failed to load or parse simulation CSV: {e}")
            return False

    def disconnect(self) -> None:
        print("\nDisconnected from simulation.")

    def get_data(self) -> dict:
        """
        Gets data from the CSV row and dynamically builds the feature vector.
        """
        if self.current_step_index >= self.total_steps:
            self.current_step_index = self.total_steps - 1 

        row = self.data.iloc[self.current_step_index]
        
        features_dict = {name: row[name] for name in self.selected_features}
        
        detection_label = int(row['label'])
        localization_label = self.ground_truth_link if detection_label == 1 else 0
        
        q_cols = [col for col in self.data.columns if col.startswith('q') and not col.startswith('q_d')]
        q_values = list(row[q_cols].values)

        self.current_step_index += 1

        return {
            "features": features_dict,
            "label": detection_label,
            "contact_link": localization_label,
            "q": q_values
        }

    def send_action(self, action_data: dict):
        self._is_performing_action = True 

    def is_performing_action(self) -> bool:
        if self.current_step_index >= self.total_steps:
            self._is_performing_action = False
        return self._is_performing_action

    def stop(self) -> None:
        self._is_performing_action = False

    def open_gripper(self):
        print("Simulation: Opening gripper.")

    def close_gripper(self):
        print("Simulation: Closing gripper.")


class FrankaRobot(RobotInterface):
    """
    Updated implementation for 'franky-control'.
    """
    def __init__(self, ip_address: str, selected_features: List[str], robot_name: str = "FrankaRobot",set_dynamic_rel: float = 0.05):
        if franky is None:
            raise ImportError("The 'franky-control' library is not installed. Please run: pip install franky-control")
        
        self.ip_address = ip_address
        self.selected_features = selected_features
        self.robot: Optional[franky.Robot] = None
        self.gripper: Optional[franky.Gripper] = None
        self.motion_thread: Optional[Thread] = None
        self.name = robot_name
        self.set_dynamic_rel = set_dynamic_rel
        
        self.latest_state: Optional[Dict] = None
        self.state_lock = threading.Lock()
        self.state_reader_thread: Optional[Thread] = None
        self.run_state_reader = False
        self.robot_lock = threading.Lock()
        self._stop_motion = False
        
        logging.info(f"🤖 FrankaRobot (using franky) initialized for IP: {self.ip_address}")

    def _state_reader_loop(self):
        """A background thread that ONLY reads robot state."""
        self.run_state_reader = True
        while self.run_state_reader:
            try:
                with self.robot_lock:
                    state_raw = self.robot.state
                
                if state_raw:
                    state_data = {
                        "q": np.array(state_raw.q),
                        "q_d": np.array(state_raw.q_d),
                        "tau_J": np.array(state_raw.tau_J)
                    }
                    state_data["e"] = state_data["q_d"] - state_data["q"]
                
                    with self.state_lock:
                        self.latest_state = state_data
                        
            except Exception as e:
                if self.run_state_reader:
                    # This can be spammy, so use debug level
                    logging.debug(f"State reader thread info: {e}") 
            time.sleep(0.001) # Read at 1000 Hz

    def connect(self) -> bool:
        logging.info(f"Connecting to Franka robot at {self.ip_address}...")
        try:
            self.robot = franky.Robot(self.ip_address)
            self.gripper = franky.Gripper(self.ip_address)
            
            self.robot.recover_from_errors()
            self.robot.relative_dynamics_factor = self.set_dynamic_rel
            
            # --- THIS IS THE FIX ---
            # Log the float value, not the object
            logging.info(f"Robot dynamics set to {self.set_dynamic_rel * 100}%.")
            # --- END FIX ---
            
            self.state_reader_thread = Thread(target=self._state_reader_loop)
            self.state_reader_thread.daemon = True
            self.state_reader_thread.start()
            
            return True
        except Exception as e:
            logging.error(f"Failed to connect to Franka robot: {e}", exc_info=True)
            return False

    def disconnect(self) -> None:
        self.run_state_reader = False
        if self.state_reader_thread and self.state_reader_thread.is_alive():
            self.state_reader_thread.join()

        if self.motion_thread and self.motion_thread.is_alive():
            self._stop_motion = True
            self.motion_thread.join()

        if self.robot:
            self.stop()
            logging.info("Disconnected from Franka robot.")
        self.robot = None

    def get_data(self) -> dict:
        """
        Gets the latest state and dynamically builds the feature vector
        based on 'self.selected_features'.
        """
        with self.state_lock:
            state = self.latest_state if self.latest_state else None
        
        if state is None:
            return {}

        features_dict = {}
        for feature_name in self.selected_features:
            match = re.match(r'([a-zA-Z_]+)(\d+)', feature_name) 
            
            if match:
                feature_base, index_str = match.groups()
                index = int(index_str)
                
                if feature_base in state and index < len(state[feature_base]):
                    features_dict[feature_name] = state[feature_base][index]
                else:
                    logging.warning(f"Feature '{feature_name}' not found in robot state keys '{list(state.keys())}' or index is out of bounds.")
            elif feature_name in state: 
                 features_dict[feature_name] = state[feature_name]
            else:
                logging.warning(f"Feature '{feature_name}' not found in robot state keys '{list(state.keys())}'.")
        
        return {"features": features_dict, "label": 0, "contact_link": 0, "q": np.array(state.get("q", []))}

    def _move(self, motion: 'franky.Motion'):
        """
        Executes the blocking move command.
        This runs in a separate thread and will be interrupted by self.robot.stop().
        """
        try:
            with self.robot_lock:
                # This is a BLOCKING call.
                # It will run until the move is done OR self.robot.stop() is called.
                self.robot.move(motion) 
            
            if not self._stop_motion:
                 logging.info("Target pose reached.")

        except Exception as e:
            # This exception is *expected* if self.robot.stop() is called
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
                motion = franky.JointMotion(target_joints)
                # We run the BLOCKING _move function in a new thread
                self.motion_thread = Thread(target=self._move, args=(motion,))
                self.motion_thread.start()
            else:
                logging.warning(f"Invalid 'target_joints' for move command.")
        elif command == "wait":
            duration = action_data.get("duration")
            if isinstance(duration, (int, float)) and duration > 0:
                self.motion_thread = Thread(target=self._wait, args=(duration,))
                self.motion_thread.start()
            else:
                logging.warning(f"Invalid 'duration' for wait command: {duration}")
        elif command == "open_gripper":
            self.open_gripper()
        elif command == "close_gripper":
            self.close_gripper()
        else:
            logging.warning(f"Command '{command}' is not yet implemented.")

    def is_performing_action(self) -> bool:
        """Checks if the motion-monitoring thread is still alive."""
        # --- THIS IS THE FIX ---
        # The only thing we need to check is if our thread is running.
        is_thread_alive = self.motion_thread is not None and self.motion_thread.is_alive()
        return is_thread_alive
        # --- END FIX ---

    def open_gripper(self, width: float = 0.08, speed: float = 0.1):
        if self.gripper:
            self.gripper.move(width, speed)
            logging.info("Gripper opened.")

    def close_gripper(self):
        if self.gripper:
            self.gripper.grasp()
            logging.info("Gripper closed.")

    def jog(self, relative_pose: List[float]):
        """Performs a small, relative motion for jogging."""
        if self.robot:
            motion = franky.LinearRelativeMotion(franky.Affine(*relative_pose))
            with self.robot_lock:
                self.robot.move(motion)
                
    def stop(self) -> None:
        """Stops any ongoing robot motion and signals threads to exit."""
        self._stop_motion = True

        if self.robot:
            logging.info("🛑 Halting robot motion.")
            self.robot.stop() # This will interrupt the blocking .move() in the thread

        if self.motion_thread and self.motion_thread.is_alive():
            self.motion_thread.join()


class URRobot(RobotInterface):
    """
    Interface for UR robots using rtde_control and rtde_receive.
    """
    def __init__(self, ip_address: str, frequency: int = 200, selected_features: Optional[List[str]] = None, robot_name: str = "URRobot"):
        self.ip_address = ip_address
        self.frequency = frequency
        self.selected_features = selected_features if selected_features is not None else []
        self.name = robot_name
        
        self.robot_control: Optional[RTDEControl] = None
        self.robot_receive: Optional[RTDEReceive] = None
        
        self.latest_state: Optional[Dict] = None
        self.state_lock = threading.Lock()
        self.robot_lock = threading.Lock()
        self.state_reader_thread: Optional[Thread] = None
        self.run_state_reader = False
        
        self._action_thread: Optional[Thread] = None
        self._stop_action = False

        logging.info(f"🤖 URRobot initialized for IP address: {self.ip_address}")

    def _state_reader_loop(self):
        """A background thread that continuously reads the robot's state."""
        self.run_state_reader = True
        logging.info("URRobot state reader thread started.")
        while self.run_state_reader:
            try:
                state_data = {
                    "q": np.array(self.robot_receive.getActualQ()),
                    "q_d": np.array(self.robot_receive.getTargetQ()),
                    "tau_J": np.array(self.robot_receive.getTargetMoment()),
                }
                state_data["e"] = state_data["q_d"] - state_data["q"]
                
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
        Gets the latest state and dynamically builds the feature vector
        based on 'self.selected_features'.
        """
        with self.state_lock:
            state = self.latest_state
        
        if state is None:
            return None

        features_dict = {}
        for feature_name in self.selected_features:
            match = re.match(r'([a-zA-Z_]+)(\d+)', feature_name) 
            
            if match:
                feature_base, index_str = match.groups()
                index = int(index_str)
                
                if feature_base in state and index < len(state[feature_base]):
                    features_dict[feature_name] = state[feature_base][index]
                else:
                    logging.warning(f"Feature '{feature_name}' not found in robot state keys '{list(state.keys())}' or index is out of bounds.")
            elif feature_name in state: 
                 features_dict[feature_name] = state[feature_name]
            else:
                logging.warning(f"Feature '{feature_name}' not found in robot state keys '{list(state.keys())}'.")
    
        return {"features": features_dict, "label": 0, "contact_link": 0, "q": np.array(state.get("q", []))}

    def send_action(self, action_data: Dict):
        if not self.run_state_reader or not self.robot_control:
            logging.error("Cannot send action: Robot is not connected.")
            return

        command = action_data.get("command")
        
        if command == "move":
            target_joint = action_data.get("joints_positions")
            speed = action_data.get("speed", 0.2)
            accel = action_data.get("acceleration", 0.1)
            
            self._stop_action = False
            self._action_thread = Thread(
                target=self._execute_move, 
                args=(self.robot_control.moveJ, target_joint, speed, accel, False) # async=False
            )
            self._action_thread.start()

        elif command == "wait":
            duration = action_data.get("duration", 1.0)
            self._stop_action = False
            self._action_thread = Thread(target=self._execute_wait, args=(duration,))
            self._action_thread.start()
        else:
            logging.warning(f"Command '{command}' is not supported by URRobot.")

    def _execute_move(self, move_function, *args):
        """
        Executes a blocking move command in a separate thread.
        """
        try:
            with self.robot_lock:
                move_function(*args) 
            
            if not self._stop_action:
                logging.info(f"Target reached for move.")
                
        except Exception as e:
            if not self._stop_action:
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
        return self.self._action_thread is not None and self._action_thread.is_alive()    

    def open_gripper(self):
        logging.warning("URRobot does not have a gripper.")

    def close_gripper(self):
        logging.warning("URRobot does not have a gripper.")
    
    def enable_freedrive(self):
        if self.robot_control:
            self.robot_control.teachMode()
            logging.info("URRobot teaching mode enabled.")

    def disable_freedrive(self):
        if self.robot_control:
            self.robot_control.endTeachMode()
            logging.info("URRobot teaching mode disabled.")
    
    def stop(self) -> None:
        """Stops any ongoing robot motion."""
        self._stop_action = True  
        
        if self.robot_control and self.robot_control.isConnected():
            try:
                logging.info("🛑 Halting UR robot motion.")
                with self.robot_lock:
                    self.robot_control.stopJ(2.0) 
            except Exception as e:
                logging.error(f"Failed to send stop command to UR robot: {e}")
        
        if self._action_thread and self._action_thread.is_alive():
            self._action_thread.join(timeout=1)
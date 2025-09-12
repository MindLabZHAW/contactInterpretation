import numpy as np
import time
import logging
from robot_interface import RobotInterface

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import frankx
except ImportError:
    frankx = None


class SimulationRobot(RobotInterface):
    """A simulated robot that reads its state sequentially from a CSV file."""
    def __init__(self, csv_file_path: str):
        if pd is None:
            raise ImportError("The 'pandas' library is not installed. Please run 'pip install pandas' to use the simulation robot.")
        
        self.file_path = csv_file_path
        self.data = None
        self.current_step_index = 0
        self.total_steps = 0
        self._is_performing_action = False
        print(f"🤖 SimulationRobot initialized with data file: {self.file_path}")

    def connect(self) -> bool:
        print("Connecting to simulation...")
        try:
            self.data = pd.read_csv(self.file_path)
            self.total_steps = len(self.data)
            self.current_step_index = 0
            if 'label' not in self.data.columns:
                logging.error(f"Ground truth 'label' column not found in {self.file_path}.")
                return False
            print(f"✅ Simulation data loaded successfully with {self.total_steps} timesteps.")
            return True
        except FileNotFoundError:
            logging.error(f"Simulation data file not found at: {self.file_path}")
            return False
        except Exception as e:
            logging.error(f"Failed to load or parse simulation CSV: {e}")
            return False

    def disconnect(self) -> None:
        print("\nDisconnected from simulation.")
        self.data = None

    def get_data(self) -> dict:
        """
        Reads the data for the CURRENT timestep. The AI model will use this to make a
        prediction for the window ENDING at this timestep. The label from this
        same timestep is used as the ground truth for that window's prediction.
        """
        if self.current_step_index >= self.total_steps:
            self.current_step_index = self.total_steps - 1

        row = self.data.iloc[self.current_step_index]
        
        # Calculate joint position error (e_q) for the current timestep
        q = row[[f'q{i}' for i in range(7)]].values
        q_d = row[[f'q_d{i}' for i in range(7)]].values
        e_q = q_d - q
        
        # Get the label for the current timestep
        ground_truth_label = int(row['label'])
        
        # Advance the index for the next call
        self.current_step_index += 1
        
        return {"e_q": e_q, "label": ground_truth_label}

    def send_action(self, action_data: dict) -> None:
        command = action_data.get('command')
        if command in ['move', 'screw']:
            print(f"SimulationRobot: Received '{command}'. Will process all data.")
            if self.current_step_index < self.total_steps:
                self._is_performing_action = True
        else:
            print(f"SimulationRobot: Unknown command '{command}'.")

    def is_performing_action(self) -> bool:
        if self.current_step_index >= self.total_steps:
            if self._is_performing_action:
                print("\nSimulationRobot: End of data reached. Action finished.")
            self._is_performing_action = False
        return self._is_performing_action

    def stop(self) -> None:
        if self._is_performing_action:
            print("\n🚨 SimulationRobot: EMERGENCY STOP 🚨")
        self._is_performing_action = False


class FrankaRobot(RobotInterface):
    # (No changes needed for FrankaRobot)
    def __init__(self, ip_address: str):
        if frankx is None:
            raise ImportError("The 'frankx' library is not installed. Please run 'pip install frankx' to use the real robot.")
        self.robot_ip = ip_address
        self.robot = None
        self.current_step_index = 0
        self.total_steps = 1
        print(f"🤖 Franka Panda Controller initialized for IP: {self.robot_ip}")

    def connect(self) -> bool:
        print(f"Connecting to Franka Panda at {self.robot_ip}...")
        try:
            self.robot = frankx.Robot(self.robot_ip)
            self.robot.recover_from_errors()
            self.robot.set_default_behavior()
            print("✅ Franka connection successful.")
            return True
        except Exception as e:
            print(f"ERROR: Failed to connect to Franka: {e}")
            return False

    def disconnect(self) -> None:
        print("Disconnected from Franka robot.")

    def get_data(self) -> dict:
        state = self.robot.read_once()
        q = np.array(state.q)
        q_d = np.array(state.q_d)
        e_q = q_d - q
        return {"e_q": e_q}

    def send_action(self, action_data: dict) -> None:
        command = action_data.get('command')
        print(f"Franka Robot: Executing command '{command}'")
        target_pose_list = action_data.get('target')
        if not target_pose_list:
            logging.warning("No 'target' provided for move command.")
            return
        pose = frankx.Pose(target_pose_list)
        if command == 'move':
            self.robot.move(pose, speed=action_data.get('speed', 0.1), non_blocking=True)
        elif command == 'screw':
            print("Franka Robot: Executing screw motion (placeholder).")
            self.robot.move(pose, speed=action_data.get('speed', 0.02), non_blocking=True)
        self.current_step_index = 1

    def is_performing_action(self) -> bool:
        is_moving = self.robot.is_moving()
        if not is_moving:
            self.current_step_index = self.total_steps
        return is_moving

    def stop(self) -> None:
        self.robot.stop_motion()
        print("🚨 Franka Robot: EMERGENCY STOP 🚨")
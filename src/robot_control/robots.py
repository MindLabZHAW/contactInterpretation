import numpy as np
import time
import logging
from robot_control.robot_interface import RobotInterface
from typing import List, Dict # <-- ADD THIS

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import frankx
except ImportError:
    frankx = None


class SimulationRobot(RobotInterface):
    """A simulated robot that reads specified feature columns from a CSV file."""
    def __init__(self, csv_file_path: str, selected_features: List[str]): # <-- CORRECTED
 
        if pd is None:
            raise ImportError("The 'pandas' library is not installed.")
        
        self.file_path = csv_file_path
        self.selected_features = selected_features
        self.data = None
        self.current_step_index = 0
        self.total_steps = 0
        self._is_performing_action = False
        print(f"🤖 SimulationRobot initialized for data file: {self.file_path}")

    def connect(self) -> bool:
        print("Connecting to simulation...")
        try:
            full_data = pd.read_csv(self.file_path)
            
            # --------------------------
            # Calculate the starting point for the second half of the data
            start_index = len(full_data) // 2
            # Slice the dataframe to only keep the second half
            self.data = full_data.iloc[start_index:].reset_index(drop=True)
            # --------------------------

            self.total_steps = len(self.data)
            self.current_step_index = 0
            # Check if all specified feature columns exist
            required_cols = self.selected_features + ['label']
            if not all(col in self.data.columns for col in required_cols):
                logging.error(f"One or more required columns are missing in {self.file_path}.")
                return False
            print(f"✅ Simulation data loaded successfully with {self.total_steps} timesteps.")
            return True
        except Exception as e:
            logging.error(f"Failed to load or parse simulation CSV: {e}")
            return False

    def disconnect(self) -> None:
        print("\nDisconnected from simulation.")

    def get_data(self) -> dict:
        """Reads the selected features and the label for the current timestep."""
        if self.current_step_index >= self.total_steps:
            self.current_step_index = self.total_steps - 1

        row = self.data.iloc[self.current_step_index]
        feature_vector = row[self.selected_features].values
        ground_truth_label = int(row['label'])
        self.current_step_index += 1
        return {"features": feature_vector, "label": ground_truth_label}
    
    def send_action(self, action_data: dict):
        self._is_performing_action = True

    def is_performing_action(self) -> bool:
        if self.current_step_index >= self.total_steps:
            self._is_performing_action = False
        return self._is_performing_action

    def stop(self) -> None:
        self._is_performing_action = False

class FrankaRobot(RobotInterface):
    """Concrete implementation for a Franka Emika Panda robot."""
    def __init__(self, ip_address: str):
        # Implementation for a real Franka robot would go here.
        # This would involve initializing the connection using 'frankx'.
        pass

    def connect(self) -> bool:
        # Code to connect to the robot controller.
        pass

    def disconnect(self) -> None:
        # Code to close the connection.
        pass

    def get_data(self) -> dict:
        """Provides the robot's state as a feature vector."""
        state = self.robot.read_once()
        q = np.array(state.q)
        dq = np.array(state.dq)
        # Combine q and dq to create the feature vector
        feature_vector = np.concatenate((q, dq))
        return {"features": feature_vector}

    def send_action(self, action_data: dict):
        # Code to send a move command to the robot.
        pass

    def is_performing_action(self) -> bool:
        return self.robot.is_moving()

    def stop(self) -> None:
        self.robot.stop_motion()
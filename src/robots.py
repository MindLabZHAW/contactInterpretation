import numpy as np
# pip install ur-rtde frankx
#
# import rtde_control
# import rtde_receive
# import frankx
import time
import logging
from robot_interface import RobotInterface

# Import pandas for the simulation robot, handling potential import errors
try:
    import pandas as pd
except ImportError:
    pd = None

# Import frankx for the real robot, handling potential import errors
try:
    import frankx
except ImportError:
    frankx = None


class SimulationRobot(RobotInterface):
    """
    A simulated robot that reads its state sequentially from a CSV file.
    This is used for offline testing of the interpreter and AI model logic.
    """
    def __init__(self, csv_file_path: str):
        if pd is None:
            raise ImportError("The 'pandas' library is not installed. Please run 'pip install pandas' to use the simulation robot.")
        
        self.file_path = csv_file_path
        self.data = None
        self.current_step_index = 0
        self.total_steps = 0
        self._is_moving = False
        self.move_step_duration = 0 # How many data rows a 'move' command consumes
        print(f"🤖 SimulationRobot initialized with data file: {self.file_path}")

    def connect(self) -> bool:
        print("Connecting to simulation...")
        try:
            self.data = pd.read_csv(self.file_path)
            self.total_steps = len(self.data)
            self.current_step_index = 0
            print(f"✅ Simulation data loaded successfully with {self.total_steps} timesteps.")
            return True
        except FileNotFoundError:
            logging.error(f"Simulation data file not found at: {self.file_path}")
            return False
        except Exception as e:
            logging.error(f"Failed to load or parse simulation CSV: {e}")
            return False

    def disconnect(self) -> None:
        print("Disconnected from simulation.")
        self.data = None

    def get_data(self) -> dict:
        """
        Reads the next row from the CSV file and calculates the joint position error (e_q).
        """
        if self.current_step_index >= self.total_steps:
            logging.warning("End of simulation data reached. Returning last known state.")
            # Prevent an index error by staying on the last row
            self.current_step_index = self.total_steps - 1

        row = self.data.iloc[self.current_step_index]
        # Assuming 7 DoF based on column names
        q = row[[f'q_{i+1}' for i in range(7)]].values
        q_d = row[[f'q_d_{i+1}' for i in range(7)]].values
        e_q = q_d - q
        
        # Advance the data pointer for the next call
        self.current_step_index += 1
        
        return {"e_q": e_q}

    def send_move_command(self, move_data: dict) -> None:
        """
        Simulates starting a move. A move will last for a set number of data points.
        """
        command = move_data.get('command')
        print(f"SimulationRobot: Received command '{command}'. Move will consume 5 data rows.")
        if self.current_step_index < self.total_steps:
            self._is_moving = True
            # Each simulated 'move' will last for 5 calls to get_data()
            self.move_step_duration = 5 
        else:
            self._is_moving = False

    def is_moving(self) -> bool:
        """
        Checks if the simulated move is still in progress by counting down the duration.
        """
        if self._is_moving:
            self.move_step_duration -= 1
            if self.move_step_duration <= 0:
                self._is_moving = False
                print("SimulationRobot: Move finished.")
        return self._is_moving

    def stop(self) -> None:
        """Stops the simulated motion immediately."""
        self._is_moving = False
        self.move_step_duration = 0
        print("🚨 SimulationRobot: EMERGENCY STOP 🚨")


class FrankaRobot(RobotInterface):
    """
    Concrete implementation for a Franka Emika Panda robot using the 'frankx' library.
    """
    def __init__(self, ip_address: str):
        if frankx is None:
            raise ImportError("The 'frankx' library is not installed. Please run 'pip install frankx' to use the real robot.")
        self.robot_ip = ip_address
        self.robot = None
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
        """
        Reads the robot's state and calculates the joint position error (e_q).
        """
        state = self.robot.read_once()
        q = np.array(state.q)
        q_d = np.array(state.q_d)
        e_q = q_d - q
        return {"e_q": e_q}

    def send_move_command(self, move_data: dict) -> None:
        """
        Sends a non-blocking move command to the robot using frankx.
        """
        command = move_data.get('command')
        print(f"Franka Robot: Executing command '{command}'")
        
        if command == 'move':
            target_pose_list = move_data.get('target')
            if target_pose_list:
                # frankx.Pose can take a list [x, y, z, a, b, c, d] for pose with quaternion
                pose = frankx.Pose(target_pose_list)
                # The move command is non-blocking, its completion is tracked by is_moving()
                self.robot.move(pose, speed=move_data.get('speed', 0.1), non_blocking=True)
        # Add other command handlers like 'open_hand', 'screw', 'insert' here as needed

    def is_moving(self) -> bool:
        """
        Checks if the robot is currently executing a motion command.
        """
        return self.robot.is_moving()

    def stop(self) -> None:
        """Stops the robot's motion immediately."""
        self.robot.stop_motion()
        print("🚨 Franka Robot: EMERGENCY STOP 🚨")


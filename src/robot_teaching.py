import logging
import json
import os
import sys
import termios
import tty
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path to allow direct execution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.robot_control import robots
from src.robot_control.robot_interface import RobotInterface
from src.config_loader import ConfigLoader

class TeachingInterface:
    """Advanced teaching interface with jogging for robots without freedrive."""

    def __init__(self, robot: RobotInterface):
        self.robot = robot
        self.sequence: List[Dict] = []
        # Dynamically set options based on robot capabilities
        self.options = ['RECORD_WAYPOINT', 'OPEN_GRIPPER', 'CLOSE_GRIPPER']
        if "URRobot" in str(type(robot)):
            self.options = ['ENABLE_FREEDRIVE', 'DISABLE_FREEDRIVE'] + self.options
        else:
            self.options = ['JOG_ROBOT'] + self.options
            
        self.selected = 0
        self.step_id_counter = 1

    def get_key(self):
        """Get a single keypress."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)
            if key == '\x1b':
                key += sys.stdin.read(2)
            return key
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def display_menu(self):
        """Display the main selection menu."""
        os.system('clear' if os.name == 'posix' else 'cls')
        print("--- 🤖 Robot Teaching Mode ---")
        print("==============================")
        print(f"Teaching Robot: {self.robot.name}")
        print("Use ↑/↓ to select, Enter to execute, 'q' to save and quit\n")

        for i, option in enumerate(self.options):
            print(f"{'>' if i == self.selected else ' '} {option}")

        print(f"\n--- Sequence ({len(self.sequence)} steps) ---")
        for step in self.sequence[-5:]:
            print(f"  - {step.get('description', 'No description')}")

    def execute_action(self):
        """Execute the currently selected action."""
        action = self.options[self.selected]

        if action == 'ENABLE_FREEDRIVE':
            self.robot.enable_freedrive()
            print("✅ Freedrive enabled. Manually position the robot.")
            input("Press Enter to continue...")
        elif action == 'DISABLE_FREEDRIVE':
            self.robot.disable_freedrive()
            print("✅ Freedrive disabled.")
            input("Press Enter to continue...")
        elif action == 'JOG_ROBOT':
            self._jog_robot()
        elif action == 'RECORD_WAYPOINT':
            self._record_waypoint()
        elif action == 'OPEN_GRIPPER':
            self._record_gripper_step(open_gripper=True)
        elif action == 'CLOSE_GRIPPER':
            self._record_gripper_step(open_gripper=False)

    def _jog_robot(self):
        """Interactive jogging mode for robots without freedrive."""
        os.system('clear')
        print("--- Jogging Mode ---")
        print("Use the following keys to move the robot:")
        print("  W/S: +Y / -Y  | A/D: +X / -X | Q/E: +Z / -Z")
        print("Press 'b' to go back to the main menu.")
        
        while True:
            key = self.get_key()
            if key == 'b':
                break
            
            jog_map = {'w': [0, 0.01, 0], 's': [0, -0.01, 0],
                       'a': [0.01, 0, 0], 'd': [-0.01, 0, 0],
                       'q': [0, 0, 0.01], 'e': [0, 0, -0.01]}
            
            if key in jog_map and hasattr(self.robot, 'jog'):
                self.robot.jog(jog_map[key])

    def _record_waypoint(self):
        """Records the robot's current position as a 'move' step."""
        print("\n--- Recording Waypoint ---")
        state = self.robot.get_data()
        position_name = input("Enter a name for this waypoint: ")

        self.sequence.append({
            "step_id": self.step_id_counter,
            "description": f"Move to waypoint: {position_name}",
            "command": "move",
            "joints_positions": state.get('q', [])
        })
        self.step_id_counter += 1
        print(f"✅ Waypoint '{position_name}' recorded.")
        input("Press Enter to continue...")

    def _record_gripper_step(self, open_gripper: bool):
        """Records a gripper command."""
        command = "open_gripper" if open_gripper else "close_gripper"
        description = "Open Gripper" if open_gripper else "Close Gripper"
        
        if open_gripper: self.robot.open_gripper()
        else: self.robot.close_gripper()
            
        self.sequence.append({
            "step_id": self.step_id_counter,
            "description": description,
            "command": command
        })
        self.step_id_counter += 1
        print(f"✅ {description} command added.")
        input("Press Enter to continue...")

    def save_sequence(self):
        """Saves the sequence to a JSON file."""
        if not self.sequence:
            print("\nNo sequence to save.")
            return

        task_name = input("\nEnter a name for this task (e.g., 'my_pick_and_place'): ")
        file_name = f"{task_name.replace(' ', '_').lower()}.json"
        save_path = os.path.join(project_root, 'src', 'config', 'robot_tasks', file_name)

        task_data = {
            "task_name": task_name,
            "description": "A task sequence created via interactive teaching.",
            "steps": self.sequence
        }

        with open(save_path, 'w') as f:
            json.dump(task_data, f, indent=2)
        print(f"\n✅ Task successfully saved to {save_path}")

    def run(self):
        """Main teaching loop."""
        if not self.robot.connect():
            logging.error("Could not connect to the robot.")
            return

        while True:
            self.display_menu()
            key = self.get_key()

            if key == '\x1b[A':  # Up
                self.selected = (self.selected - 1) % len(self.options)
            elif key == '\x1b[B':  # Down
                self.selected = (self.selected + 1) % len(self.options)
            elif key == '\r':  # Enter
                self.execute_action()
            elif key == 'q':
                break
        
        self.save_sequence()
        self.robot.disconnect()

def select_robot_from_list(config: dict) -> Optional[str]:
    robot_options = list(config.get("robot_settings", {}).keys())
    if not robot_options:
        logging.error("No robots found in config.")
        return None

    print("\n--- Please Select a Robot to Teach ---")
    for i, name in enumerate(robot_options):
        print(f"[{i + 1}] {name}")
    
    while True:
        try:
            choice = int(input(f"Enter your choice (1-{len(robot_options)}): "))
            if 1 <= choice <= len(robot_options):
                return robot_options[choice - 1]
            else:
                print("Invalid choice.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def create_robot(config: dict, robot_name: str):
    settings = config.get("robot_settings", {}).get(robot_name)
    if not settings: return None
    try:
        RobotClass = getattr(robots, settings["class_name"])
        return RobotClass(**settings["init_args"])
    except Exception as e:
        logging.error(f"Failed to create robot '{robot_name}': {e}")
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    config_loader = ConfigLoader()
    config = config_loader.load(os.path.join(project_root, 'src', 'config', 'config.json'))
    if not config: sys.exit(1)

    robot_to_teach = select_robot_from_list(config)
    if not robot_to_teach: sys.exit(1)
        
    my_robot = create_robot(config, robot_to_teach)
    if my_robot:
        teacher = TeachingInterface(my_robot)
        teacher.run()
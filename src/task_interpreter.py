import time
from robot_interface import RobotInterface
from config_loader import ConfigLoader
class ContactDetectorAI:
    """
    A class to encapsulate the contact detection model.
    For this example, it uses a simple threshold, but you would
    load and run your actual AI model here.
    """
    def __init__(self, threshold: float = 5.0):
        #TODO: THIS CLASS SHOULD GET AI MODEL AND LOAD IT HERE.
        self.contact_threshold = threshold
        print(f"🧠 AI Contact Detector Initialized with threshold: {self.contact_threshold}N.")

    def predict_contact(self, robot_data: dict) -> bool:
        """
        Predicts contact based on robot force data.
        
        Args:
            robot_data (dict): A dictionary containing sensor data from the robot.
                               Expected to have a 'force_z' key.
        
        Returns:
            bool: True if contact is detected, False otherwise.
        """
        force = robot_data.get("force_z", 0)
        return force > self.contact_threshold

class TaskInterpreter:
    """
    The main brain of the application. It interprets a task defined in a
    JSON file, commands the robot, and uses the AI model to react to
    real-time contact events based on a prioritized set of rules.
    """
    def __init__(self, robot: RobotInterface, ai_model: ContactDetectorAI, default_contact_actions: dict = None):
        self.robot = robot
        self.ai = ai_model
        self.task_data = None
        # Store the default actions defined in main.py. If none are provided, use an empty dictionary.
        self.default_actions = default_contact_actions if default_contact_actions else {}

    def load_task_from_file(self, file_path: str):
        """Loads and stores the task definition from a JSON file."""
        self.task_data = ConfigLoader().load(file_path)

    def run(self):
        """The main execution handler for the entire loaded task."""
        if not self.task_data:
            print("ERROR: No task loaded. Call load_task_from_file() first. Aborting.")
            return

        if not self.robot.connect():
            print("ERROR: Failed to connect to robot. Aborting.")
            return

        print(f"\n--- ✅ Starting Task: {self.task_data.get('name', 'Untitled Task')} ---")
        try:
            # Execute each step from the JSON file sequentially
            for step in self.task_data.get('steps', []):
                print(f"\nExecuting Step {step.get('id', '?')}: {step.get('command', 'Unknown Command')}")
                step_successful = self.execute_step(step)
                if not step_successful:
                    print(f"--- ⚠️ Task halted due to contact event at step {step.get('id', '?')} ---")
                    break # Stop executing the rest of the task
            else: # This 'else' belongs to the 'for' loop, runs if the loop completes without 'break'
                 print("\n--- ✅ All steps completed successfully. ---")
        finally:
            self.robot.disconnect()
            print("--- ⏹️ Task Finished ---")

    def execute_step(self, step: dict) -> bool:
        """
        Executes a single step and contains the real-time monitoring loop.
        Returns True if the step completes, False if interrupted by contact.
        """
        # Send the initial command to the robot (this should be non-blocking)
        self.robot.send_move_command(step)
        
        # MONITORING LOOP: Check for contact while the robot is busy
        while self.robot.is_moving():
            robot_data = self.robot.get_data()
            if self.ai.predict_contact(robot_data):
                self.robot.stop() # Immediately stop the robot
                print(f"CONTACT DETECTED during step {step.get('id', '?')} ({step.get('command', 'Unknown')})!")
                self.handle_contact(step) 
                return False # Step was interrupted

            time.sleep(0.01) # High-frequency check (100 Hz)
        
        print(f"Step {step.get('id', '?')} completed without contact.")
        return True # Step completed without interruption

    def handle_contact(self, step: dict):
        """
        Handles a contact event with a priority-based rule system.
        Priority 1: Step-specific 'on_contact' from JSON.
        Priority 2: Command-type default action.
        Priority 3: Global fallback (stop the task).
        """
        command_type = step.get('command')
        contact_action = None

        # Priority 1: Check for a specific action in the step's JSON definition.
        if 'on_contact' in step and step['on_contact']:
            print("Handling contact with STEP-SPECIFIC action from JSON.")
            contact_action = step['on_contact']
        
        # Priority 2: If no specific action, check for a default for this command type.
        elif command_type in self.default_actions:
            print(f"Handling contact with COMMAND-DEFAULT action for '{command_type}'.")
            contact_action = self.default_actions[command_type]
            
        # Priority 3: If no other rule applies, use the safest global fallback.
        else:
            print("Handling contact with GLOBAL FALLBACK action.")
            contact_action = {
                "action": "stop_task",
                "message": f"Critical Error: No contact rule defined for command '{command_type}'."
            }
            
        # Now, execute the chosen action
        action = contact_action.get('action', 'stop_task')
        message = contact_action.get('message', 'No message.')
        print(f"--> Action: {action.upper()}. Message: {message}")

        if action == "retry":
            print("--> Logic for retrying the step would be implemented here.")
        elif action == "log_and_continue":
            print("--> Logging event and preparing to continue to the next step.")
        # For 'stop_task', no further action is needed as the loop will break.
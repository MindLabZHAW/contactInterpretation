import time
from abstract_robot import AbstractRobot

class ContactDetectorAI:
    """
    A class to encapsulate the contact detection model.
    For this example, it uses a simple threshold, but you would
    load and run your actual AI model here.
    """
    def __init__(self, threshold: float = 5.0):
        self.contact_threshold = threshold
        print(f"🧠 AI Contact Detector Initialized with threshold: {self.contact_threshold}N.")

    def predict_contact(self, robot_data: dict) -> bool:
        """
        Predicts contact based on robot data.
        
        Args:
            robot_data (dict): A dictionary containing sensor data from the robot.
                               Expected to have a 'force_z' key.
        
        Returns:
            bool: True if contact is detected, False otherwise.
        """
        force = robot_data.get("force_z", 0)
        # In a real application, you would replace this logic with:
        # processed_input = self.preprocess(robot_data)
        # prediction = self.model.predict(processed_input)
        return force > self.contact_threshold

class Orchestrator:
    """
    The main brain of the application. It orchestrates the process by
    reading data, getting AI predictions, and commanding the robot.
    It operates on any robot that adheres to the AbstractRobot interface.
    """
    def __init__(self, robot: AbstractRobot, ai_model: ContactDetectorAI):
        self.robot = robot
        self.ai = ai_model
        self.is_running = False

    def run_process(self):
        """The main real-time control loop."""
        if not self.robot.connect():
            print("ERROR: Failed to connect to robot. Aborting process.")
            return

        self.is_running = True
        print("\n--- ✅ Starting Robot Process ---")
        try:
            while self.is_running:
                # 1. Read the latest data from the robot
                current_data = self.robot.get_data()
                
                # 2. Get a prediction from the AI model
                is_contact = self.ai.predict_contact(current_data)
                
                # 3. Decide and act based on the prediction
                if is_contact:
                    print(f"CONTACT DETECTED! Force: {current_data.get('force_z', 0):.2f}N")
                    self.robot.stop()
                    self.is_running = False # Stop the loop on contact
                else:
                    print(f"No contact. Force: {current_data.get('force_z', 0):.2f}N. Continuing move.")
                    self.robot.send_move_command(move_data="next_waypoint")
                
                # Control the frequency of the loop (e.g., 10 Hz)
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nProcess interrupted by user.")
        finally:
            self.robot.disconnect()
            print("--- ⏹️ Robot Process Finished ---")
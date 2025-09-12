import time
import logging
import sys # Import sys to control the output stream
from robot_interface import RobotInterface
from contact_model import ContactDetectorAI
from config_loader import ConfigLoader
from plotting import plot_results

class TaskInterpreter:
    """
    Reads a task definition, executes it, and generates a plot
    comparing AI predictions to ground truth data from a simulation.
    Handles Ctrl+C interruption to show results early.
    """
    def __init__(self, robot: RobotInterface, ai_model: ContactDetectorAI, default_contact_actions: dict, loop_delay: float = 0.01):
        self.robot = robot
        self.ai = ai_model
        self.task = None
        self.default_contact_actions = default_contact_actions
        self.loop_delay = loop_delay
        
        self.ground_truths = []
        self.predictions = []

    def load_task_from_file(self, file_path: str):
        loader = ConfigLoader()
        self.task = loader.load(file_path)
        if self.task:
            logging.info(f"Task '{self.task.get('task_name', 'Untitled Task')}' loaded successfully.")
        else:
            logging.error(f"Failed to load task from {file_path}.")

    def run(self):
        if not self.task:
            logging.error("No task loaded. Aborting run.")
            return

        if not self.robot.connect():
            logging.error("Robot connection failed. Aborting task.")
            return
        
        try:
            logging.info(f"--- ✅ Starting Task: {self.task.get('task_name', 'Untitled Task')} ---")
            logging.info("Press Ctrl+C at any time to stop and see the plot with the data collected so far.")

            self.ground_truths = []
            self.predictions = []

            for step in self.task.get('steps', []):
                self.execute_step(step)
            
            logging.info("--- ✅ All steps completed successfully. ---")

        except KeyboardInterrupt:
            logging.warning("\n--- 🛑 Task interrupted by user (Ctrl+C) ---")
        
        finally:
            # Ensure the final log messages are on a new line
            print() 
            logging.info("--- Generating plot with collected data ---")
            plot_results(self.ground_truths, self.predictions)
            self.robot.disconnect()
            logging.info("--- ⏹️ Task Finished ---")

    def execute_step(self, step: dict):
        """Executes a single step and collects data while showing progress."""
        logging.info(f"Executing Step {step['step_id']}: {step['command']}")
        self.robot.send_action(step)
        
        while self.robot.is_performing_action():
            # --- Progress Bar Logic ---
            total = self.robot.total_steps
            if total > 1: # Only show progress for simulations
                current = self.robot.current_step_index
                percent = (current / total) * 100
                # Use sys.stdout to write on a single line
                progress_bar = f"Processing data: {current}/{total} ({percent:.1f}%)"
                sys.stdout.write(f"\r{progress_bar}")
                sys.stdout.flush()
            # --------------------------

            robot_data = self.robot.get_data()
            
            prediction_bool = self.ai.predict_contact(robot_data)
            prediction_int = 1 if prediction_bool else 0
            
            try:
                ground_truth_int = robot_data["label"]
            except KeyError:
                logging.error("\nFatal: 'label' key not found in robot_data.")
                self.robot.stop() 
                break

            self.predictions.append(prediction_int)
            self.ground_truths.append(ground_truth_int)

            if self.loop_delay > 0:
                time.sleep(self.loop_delay)

        # After the loop, print a newline to move past the progress bar
        print() 
        logging.info(f"Step {step.get('step_id', '?')} data processing complete.")
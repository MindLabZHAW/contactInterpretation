import time
import logging
import sys
from collections import Counter
import pandas as pd
from typing import Dict, List, Optional
import os # <-- Added for path joining

# Import from your custom packages
from robot_control.robot_interface import RobotInterface
from contact_interpretation.interpreter import ContactAI
from config_loader import ConfigLoader
from contact_interpretation.plotting import plot_results, RealTimePlotter
from data_logger import DataLogger

# --- Path Setup ---
# This ensures paths in configs are relative to the project root (contactInterpretation)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class TaskInterpreter:
    def __init__(self, robot: RobotInterface, ai_model: ContactAI, default_contact_actions: Dict, data_logger: Optional[DataLogger] = None, loop_delay: float = 0.01, realtime_plot: bool = False):
        self.robot = robot
        self.ai = ai_model
        self.task = None
        self.default_contact_actions = default_contact_actions
        if data_logger:
            data_logger.robot_name= self.robot.name
        self.data_logger = data_logger
        self.loop_delay = loop_delay
        # Initialize the real-time plotter if requested
        self.plotter = RealTimePlotter(robot_name = self.robot.name) if realtime_plot else None

        self.detection_ground_truths: List[int] = []
        self.raw_predictions: List[int] = []

        self.smoothed_predictions: List[int] = []
        self.localization_predictions: List[int] = []

    def load_task_from_file(self, file_path: str):
        """
        Loads a JSON task file.
        Paths in the config.yaml might be relative to the project root.
        """
        loader = ConfigLoader()
        
        # Ensure the path is absolute
        if not os.path.isabs(file_path):
            file_path = os.path.join(project_root, file_path)
            
        # --- THIS IS THE UPDATE ---
        # We now explicitly call load_json, as ConfigLoader is no longer generic.
        self.task = loader.load_json(file_path)
        # ------------------------
        
        if self.task:
            logging.info(f"Task '{self.task.get('task_name', 'Untitled Task')}' loaded successfully.")
        else:
            logging.error(f"Failed to load task from {file_path}.")

    def run(self):
        if not self.task or not self.robot.connect():
            logging.error("Setup failed. Aborting run.")
            return

        try:
            logging.info(f"--- ✅ Starting Task: {self.task.get('task_name', 'Untitled Task')} ---")
            logging.info("Press Ctrl+C at any time to stop and see the plot with the data collected so far.")
            
            if self.data_logger:
                self.data_logger.start()
            
            # Start the plotting process if it exists
            if self.plotter:
                self.plotter.start()

            self.detection_ground_truths = []
            self.raw_predictions = []
            self.smoothed_predictions = []
            self.localization_predictions = []

            for step in self.task.get('steps', []):
                if not self.execute_step(step):
                    # If execute_step returns False (due to a "stop_task" action), break the loop.
                    logging.warning(f"--- 🛑 Task halted by contact behavior. ---")
                    break

            logging.info("--- ✅ All steps completed successfully. ---")

        except KeyboardInterrupt:
            logging.warning("\n--- 🛑 Task interrupted by user (Ctrl+C) ---")

        finally:
            # Close the real-time plot window if it exists
            if self.plotter:
                self.plotter.close()
            self.robot.disconnect()

            try:
                logging.info("--- Generating plot with collected data ---")
                plot_results(
                    ground_truth=self.detection_ground_truths,
                    predictions=self.raw_predictions,
                    smoothed_predictions=self.smoothed_predictions,
                    localization_predictions=self.localization_predictions,
                    robot_name=self.robot.name
                )
            except KeyboardInterrupt:
                logging.warning("Final plot interrupted by user. Skipping plot generation.")

            if self.data_logger:
                self.data_logger.save()

            logging.info("--- ⏹️ Task Finished ---")

    def execute_step(self, step: Dict) -> bool:
        """
        Executes a single step and runs the monitoring loop.
        Returns False if the task should be stopped, True otherwise.
        """
        step_command = step.get('command', 'N/A')
        logging.info(f"Executing Step {step.get('step_id', '?')}: {step_command}")
        self.robot.send_action(step)
        time.sleep(0.1) # Give the robot a moment to start the action
        
        # --- Main Monitoring Loop ---
        while self.robot.is_performing_action():
            start_time = time.time()
            robot_data = self.robot.get_data()
            
            if robot_data:
                raw_pred, smoothed_pred, loc_pred = self.ai.predict(robot_data)
                
                # Store data for plotting
                self.raw_predictions.append(raw_pred)
                self.smoothed_predictions.append(smoothed_pred)
                self.localization_predictions.append(loc_pred)
                ground_truth_label = robot_data.get("label", 0)
                self.detection_ground_truths.append(ground_truth_label)

                # Update plotter (if enabled)
                if self.plotter:
                    self.plotter.update(ground_truth_label, raw_pred, smoothed_pred, loc_pred)

                # Log data (if enabled)
                if self.data_logger:
                    log_entry = {
                        'label': ground_truth_label,
                        'raw_pred': raw_pred,
                        'smoothed_pred': smoothed_pred,
                        'loc_pred': loc_pred,
                        **robot_data.get('features', {})
                    }
                    self.data_logger.log(log_entry)
                
                # --- NEW: Contact Handling Logic ---
                if smoothed_pred == 1:
                    # A contact was detected!
                    if not self.handle_contact(step):
                        # The handling function returned False, meaning "stop the task"
                        self.robot.stop()
                        return False # Stop executing steps

            # --- Loop Delay Compensation ---
            diff = time.time() - start_time
            time.sleep(self.loop_delay - diff if self.loop_delay > diff else 0)

        logging.info(f"Step {step.get('step_id', '?')} data processing complete.")
        return True # Continue to the next step

    def handle_contact(self, step: Dict) -> bool:
        """
        Determines the correct action to take based on a contact event.
        Returns False if the task should stop, True otherwise.
        """
        command = step.get("command", "N/A")
        
        # 1. Check for a specific "on_contact" rule in the current step
        step_behavior = step.get("on_contact")
        
        # 2. If no specific rule, check for a default rule for this command
        default_behavior = self.default_contact_actions.get(command)
        
        # 3. Use the most specific behavior found (step > default)
        behavior = step_behavior or default_behavior
        
        if not behavior:
            # 4. If no rule is found, use a "safe" failsafe
            logging.error(f"Contact detected during '{command}' but no 'on_contact' or default behavior was found. Stopping task for safety.")
            return False # Stop task

        # --- Execute the determined action ---
        action = behavior.get("action", "stop_task")
        message = behavior.get("message", "No message provided.")

        if action == "stop_task":
            logging.warning(f"Contact detected: {message}. Action: Stopping task.")
            return False # Signal to stop
            
        elif action == "log_and_continue":
            logging.info(f"Contact detected: {message}. Action: Logging and continuing.")
            return True # Signal to continue
            
        elif action == "retry":
            # (Note: This is a simplified "retry". A real implementation
            # would need more complex logic, e.g., move up, then try again.)
            logging.warning(f"Contact detected: {message}. Action: Retrying step.")
            # For now, "retry" will just log and continue.
            return True # Signal to continue

        else:
            logging.warning(f"Unknown action '{action}' defined. Defaulting to 'stop_task'.")
            return False # Signal to stop
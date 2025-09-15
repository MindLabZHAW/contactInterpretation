import time
import logging
import sys
from collections import Counter
import pandas as pd
from typing import Dict, List

# Import from your custom packages
from robot_control.robot_interface import RobotInterface
from contact_interpretation.interpreter import ContactAI
from config_loader import ConfigLoader
from contact_interpretation.plotting import plot_results

class TaskInterpreter:
    """
    Reads a task definition, executes it by controlling a robot,
    and uses an AI model to interpret contact events in real-time.
    """
    def __init__(self, robot: RobotInterface, ai_model: ContactAI, default_contact_actions: Dict, loop_delay: float = 0.01):
        self.robot = robot
        self.ai = ai_model
        self.task = None
        self.default_contact_actions = default_contact_actions
        self.loop_delay = loop_delay

        # Lists to store data for final plotting
        self.detection_ground_truths: List[int] = []
        self.raw_predictions: List[int] = [] # For raw model output
        self.smoothed_predictions: List[int] = [] # For smoothed output
        self.localization_predictions: List[int] = []

    def load_task_from_file(self, file_path: str):
        """Loads a task sequence from a JSON file."""
        loader = ConfigLoader()
        self.task = loader.load(file_path)
        if self.task:
            logging.info(f"Task '{self.task.get('task_name', 'Untitled Task')}' loaded successfully.")
        else:
            logging.error(f"Failed to load task from {file_path}.")

    def run(self):
        """Starts the main execution loop for the loaded task."""
        if not self.task:
            logging.error("No task loaded. Aborting run.")
            return

        if not self.robot.connect():
            logging.error("Robot connection failed. Aborting task.")
            return

        try:
            logging.info(f"--- ✅ Starting Task: {self.task.get('task_name', 'Untitled Task')} ---")
            logging.info("Press Ctrl+C at any time to stop and see the plot with the data collected so far.")

            # Clear previous run data
            self.detection_ground_truths = []
            self.raw_predictions = []
            self.smoothed_predictions = []
            self.localization_predictions = []

            for step in self.task.get('steps', []):
                self.execute_step(step)

            logging.info("--- ✅ All steps completed successfully. ---")

        except KeyboardInterrupt:
            logging.warning("\n--- 🛑 Task interrupted by user (Ctrl+C) ---")

        finally:
            print() 
            logging.info("--- Generating plot with collected data ---")

            plot_results(
                ground_truth=self.detection_ground_truths,
                predictions=self.raw_predictions, # Pass the raw predictions
                smoothed_predictions=self.smoothed_predictions, # Pass the smoothed predictions
                localization_predictions=self.localization_predictions
            )

            self.robot.disconnect()
            logging.info("--- ⏹️ Task Finished ---")

    def execute_step(self, step: Dict):
        """Executes a single step of the task and collects data."""
        logging.info(f"Executing Step {step.get('step_id', '?')}: {step.get('command', 'N/A')}")
        self.robot.send_action(step)

        while self.robot.is_performing_action():
            # ... (Progress bar logic remains the same) ...
            if hasattr(self.robot, 'total_steps') and self.robot.total_steps > 1:
                current = self.robot.current_step_index
                total = self.robot.total_steps
                percent = (current / total) * 100
                progress_bar = f"\rProcessing data: {current}/{total} ({percent:.1f}%)"
                sys.stdout.write(progress_bar)
                sys.stdout.flush()

            robot_data = self.robot.get_data()
            if not robot_data:
                continue

            # Get all three outputs from the AI
            raw_pred, smoothed_pred, loc_pred = self.ai.predict(robot_data)

            try:
                ground_truth_int = robot_data["label"]
            except KeyError:
                logging.error("\nFatal: 'label' key not found in robot_data. Cannot get ground truth.")
                self.robot.stop()
                break

            # Append new data to our lists
            self.raw_predictions.append(raw_pred)
            self.smoothed_predictions.append(smoothed_pred)
            self.localization_predictions.append(loc_pred)
            self.detection_ground_truths.append(ground_truth_int)

            if self.loop_delay > 0:
                time.sleep(self.loop_delay)

        if hasattr(self.robot, 'total_steps') and self.robot.total_steps > 1:
            print()
        logging.info(f"Step {step.get('step_id', '?')} data processing complete.")
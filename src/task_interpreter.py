import time
import logging
import sys
from collections import Counter
import pandas as pd
from typing import Dict, List, Optional

# Import from your custom packages
from robot_control.robot_interface import RobotInterface
from contact_interpretation.interpreter import ContactAI
from config_loader import ConfigLoader
from contact_interpretation.plotting import plot_results, RealTimePlotter
from data_logger import DataLogger

class TaskInterpreter:
    def __init__(self, robot: RobotInterface, ai_model: ContactAI, default_contact_actions: Dict, data_logger: Optional[DataLogger] = None, loop_delay: float = 0.01, realtime_plot: bool = False):
        self.robot = robot
        self.ai = ai_model
        self.task = None
        self.default_contact_actions = default_contact_actions
        self.data_logger = data_logger
        self.loop_delay = loop_delay
        # Initialize the real-time plotter if requested
        self.plotter = RealTimePlotter() if realtime_plot else None

        self.detection_ground_truths: List[int] = []
        self.raw_predictions: List[int] = []

        self.smoothed_predictions: List[int] = []
        self.localization_predictions: List[int] = []

    def load_task_from_file(self, file_path: str):
        loader = ConfigLoader()
        self.task = loader.load(file_path)
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
                self.execute_step(step)

            logging.info("--- ✅ All steps completed successfully. ---")

        except KeyboardInterrupt:
            logging.warning("\n--- 🛑 Task interrupted by user (Ctrl+C) ---")

        finally:
            # Close the real-time plot window if it exists
            if self.plotter:
                self.plotter.close()

            logging.info("--- Generating plot with collected data ---")
            plot_results(
                ground_truth=self.detection_ground_truths,
                predictions=self.raw_predictions,
                smoothed_predictions=self.smoothed_predictions,
                localization_predictions=self.localization_predictions
            )
            
            if self.data_logger:
                self.data_logger.save()

            self.robot.disconnect()
            logging.info("--- ⏹️ Task Finished ---")

    def execute_step(self, step: Dict):
        """Executes a single step and runs the monitoring loop."""
        logging.info(f"Executing Step {step.get('step_id', '?')}: {step.get('command', 'N/A')}")
        self.robot.send_action(step)
        time.sleep(0.1)
        

        while self.robot.is_performing_action():
            start_time = time.time()
            robot_data = self.robot.get_data()
            
            if robot_data:
                raw_pred, smoothed_pred, loc_pred = self.ai.predict(robot_data)
                #diff = time.time() - start_time
                #logging.info(f'diff: {diff:.4f}s, loop_delay: {self.loop_delay:.4f}s')
                # Store data for plotting
                self.raw_predictions.append(raw_pred)
                self.smoothed_predictions.append(smoothed_pred)
                self.localization_predictions.append(loc_pred)
                ground_truth_label = robot_data.get("label", 0)
                self.detection_ground_truths.append(ground_truth_label)

                if self.plotter:
                    self.plotter.update(ground_truth_label, raw_pred, smoothed_pred, loc_pred)

                if self.data_logger:
                    log_entry = {
                        'label': ground_truth_label,
                        'raw_pred': raw_pred,
                        'smoothed_pred': smoothed_pred,
                        'loc_pred': loc_pred,
                        **robot_data.get('features', {})
                    }
                    self.data_logger.log(log_entry)
            diff = time.time() - start_time
            time.sleep(self.loop_delay-diff if self.loop_delay > diff else 0)
            '''if self.loop_delay < diff:
                logging.info(f'diff: {diff:.4f}s, smoothed_pred: {smoothed_pred}, loc_pred: {loc_pred}')'''

        logging.info(f"Step {step.get('step_id', '?')} data processing complete.")
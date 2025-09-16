import pandas as pd
import time
import logging
from typing import List, Dict
import os

class DataLogger:
    """
    A class to log real-time robot and AI data and save it to a CSV file.
    """
    def __init__(self, headers: List[str]):
        """
        Initializes the logger with the specified column headers.
        """
        # Prepend 'timestamp' to the list of headers you provide
        self.headers = ['timestamp'] + headers
        self.data_rows: List[list] = []
        self.start_time = None

        # Create a 'logs' directory if it doesn't already exist
        self.log_directory = 'logs'
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)
            logging.info(f"Created logging directory at: {self.log_directory}")

    def start(self):
        """Starts the timer for the logging session."""
        self.start_time = time.time()
        logging.info("Data logger started.")

    def log(self, data_dict: Dict):
        """
        Logs a new row of data.

        Args:
            data_dict (Dict): A dictionary containing the data for the current timestep.
                              Keys should match the headers provided during initialization.
        """
        if self.start_time is None:
            logging.warning("Logger has not been started. Call start() before logging.")
            return
            
        # Calculate elapsed time since the start of the run
        timestamp = time.time() - self.start_time
        # Create a row of data in the correct order based on the headers
        row = [timestamp] + [data_dict.get(h) for h in self.headers[1:]]
        self.data_rows.append(row)

    def save(self):
        """Saves all logged data to a timestamped CSV file."""
        if not self.data_rows:
            logging.info("No data was logged, skipping save.")
            return

        # Create a unique, timestamped filename (e.g., contact_data_20250916-103055.csv)
        timestamp_str = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.log_directory, f"contact_data_{timestamp_str}.csv")

        # Use pandas to create a DataFrame and save it to a CSV file
        df = pd.DataFrame(self.data_rows, columns=self.headers)
        df.to_csv(filename, index=False)
        logging.info(f"✅ Data successfully saved to {filename}")
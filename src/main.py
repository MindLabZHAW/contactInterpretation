import logging
from task_interpreter import TaskInterpreter
from contact_model import ContactDetectorAI
from robots import FrankaRobot, SimulationRobot # Import the new SimulationRobot
from config_loader import ConfigLoader

if __name__ == "__main__":
    """
    Main entry point for the robot task interpreter application.
    This script is fully configuration-driven and can be run in
    either live mode with a real robot or in simulation mode using CSV data.
    """
    logging.info("--- Initializing System ---")
    
    # --- 1. Initialize AI Model ---
    # This will prompt the user in the console to select a model.
    contact_ai = ContactDetectorAI()
    if contact_ai.model is None:
        logging.error("Model loading failed. Exiting application.")
        exit()

    # --- 2. Select Robot Implementation: LIVE or SIMULATION ---
    
    # --- Option A: Live Mode with a real Franka robot ---
    # my_robot = FrankaRobot(ip_address="192.168.1.15") 

    # --- Option B: Simulation Mode using data from a CSV file ---
    my_robot = SimulationRobot(csv_file_path='src/config/simulated_robot_data.csv')

    # --- 3. Load Configuration Files ---
    config_loader = ConfigLoader()
    default_behaviors = config_loader.load('src/config/default_behaviors.json')
    if not default_behaviors:
        logging.warning("Could not load default behaviors. Continuing with no defaults.")
        default_behaviors = {}

    # --- 4. Configure and Run the TaskInterpreter ---
    controller = TaskInterpreter(
        robot=my_robot, 
        ai_model=contact_ai, 
        default_contact_actions=default_behaviors
    )
    
    controller.load_task_from_file('src/config/my_assembly_task.json')
    
    controller.run()
    
    logging.info("--- System Shutdown ---")


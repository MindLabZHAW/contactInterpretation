import logging
from task_interpreter import TaskInterpreter
from contact_interpretation.interpreter import ContactAI
from robot_control.robots import FrankaRobot, SimulationRobot
from config_loader import ConfigLoader

if __name__ == "__main__":
    """
    Main entry point for the robot contact interpretation application.
    This script ties together the robot control and contact interpretation packages.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("--- Initializing System ---")

    # --- 1. Define Model and Feature Configuration ---
    # The feature columns the AI models were trained on. The order must be consistent.
    SELECTED_FEATURES = [
        'e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6'
    ]
    NUM_FEATURES = len(SELECTED_FEATURES)

    # --- 2. Initialize AI Models ---
    # This will trigger the interactive prompts for model selection.
    contact_ai = ContactAI(num_features=NUM_FEATURES)
    if contact_ai.detection_model is None or contact_ai.localization_model is None:
        logging.error("Model loading failed. Exiting application.")
        exit()

    # --- 3. Select and Initialize Robot Implementation ---
    # For this example, we use the SimulationRobot.
    default_csv_path = 'dataset/franka_main/labeled_data/link5/c4_1.csv'
    csv_path_input = input(f"Enter the path to the simulation CSV file [{default_csv_path}]: ")
    # Use the default path if the user just presses Enter
    if not csv_path_input:
        csv_path_input = default_csv_path
        
    my_robot = SimulationRobot(
        csv_file_path=csv_path_input,
        selected_features=SELECTED_FEATURES
    )

    # --- 4. Load Configuration Files ---
    config_loader = ConfigLoader()
    # Note: The path is relative to the project root where the script is run from.
    default_behaviors = config_loader.load('src/config/default_behaviors.json')
    if not default_behaviors:
        logging.warning("Could not load default behaviors. Continuing with no defaults.")
        default_behaviors = {}

    # --- 5. Configure and Run the Task Interpreter ---
    # The controller brings all the components together.
    controller = TaskInterpreter(
        robot=my_robot, 
        ai_model=contact_ai, 
        default_contact_actions=default_behaviors,
        loop_delay=0  # Set to 0 for fastest simulation, or >0 to slow it down.
    )
    
    controller.load_task_from_file('src/config/my_assembly_task.json')
    
    # This starts the main application loop.
    controller.run()
    
    logging.info("--- System Shutdown ---")
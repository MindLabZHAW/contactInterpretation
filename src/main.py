import logging
import socket
import sys
from task_interpreter import TaskInterpreter
from contact_interpretation.interpreter import ContactAI
from robot_control import robots 
from config_loader import ConfigLoader
from data_logger import DataLogger

def select_robot_from_list(config: dict) -> str:
    """
    Displays a menu of available robots from the config file and prompts the user to choose one.
    """
    robot_options = list(config.get("robot_settings", {}).keys())
    if not robot_options:
        logging.error("No robots found in the 'robot_settings' of your config file.")
        return None

    print("\n--- Please Select a Robot ---")
    for i, name in enumerate(robot_options):
        print(f"[{i + 1}] {name}")
    
    while True:
        try:
            choice = int(input(f"Enter your choice (1-{len(robot_options)}): "))
            if 1 <= choice <= len(robot_options):
                return robot_options[choice - 1]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def discover_active_robot(config: dict) -> str:
    """
    Tries to connect to each robot defined in the config to find which one is active.
    Returns the name of the first robot that responds.
    """
    robot_settings = config.get("robot_settings", {})
    for name, settings in robot_settings.items():
        ip = settings.get("init_args", {}).get("ip_address")
        if not ip:
            continue 

        try:
            logging.info(f"Checking for robot '{name}' at {ip}...")
            with socket.create_connection((ip, 80), timeout=1):
                logging.info(f"✅ Found active robot: {name}")
                return name
        except (socket.timeout, ConnectionRefusedError):
            logging.info(f"No response from '{name}'.")
            continue
            
    return None


def create_robot(config: dict, active_robot_name: str):
    """
    Factory function to dynamically create a robot instance from configuration.
    """
    settings = config.get("robot_settings", {}).get(active_robot_name)

    if not settings:
        logging.error(f"Configuration for active robot '{active_robot_name}' not found.")
        return None, None, None, None

    class_name = settings.get("class_name")
    init_args = settings.get("init_args", {})
    selected_features = init_args.get("selected_features")
    
    try:
        RobotClass = getattr(robots, class_name)
        robot_instance = RobotClass(**init_args)
    except AttributeError:
        logging.error(f"Robot class '{class_name}' not found in 'robot_control.robots' module.")
        return None, None, None, None
    except Exception as e:
        logging.error(f"Failed to instantiate robot '{class_name}': {e}")
        return None, None, None, None

    task_name = settings.get("task_name")
    loop_delay = settings.get("loop_delay")
    
    return robot_instance, task_name, loop_delay, selected_features

if __name__ == "__main__":
    """
    Main entry point for the robot contact interpretation application.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("--- Initializing System ---")
    
    config_loader = ConfigLoader()
    config = config_loader.load('src/config/config.json')
    if not config:
        logging.error("Could not load config.json. Exiting application.")
        sys.exit(1)

    active_robot_name = config.get("active_robot")
    
    if active_robot_name == "INTERACTIVE":
        active_robot_name = select_robot_from_list(config)
        if not active_robot_name:
            sys.exit(1)
    elif active_robot_name == "AUTO_DETECT":
        active_robot_name = discover_active_robot(config)
        if not active_robot_name:
            logging.error("Auto-detect failed: No configured robots found on the network.")
            sys.exit(1)
    elif active_robot_name not in config.get("robot_settings", {}):
        logging.error(f"The specified active_robot '{active_robot_name}' does not exist in the configuration.")
        sys.exit(1)
        
    my_robot, task_name, loop_delay, selected_features = create_robot(config, active_robot_name)
    
    if my_robot is None:
        sys.exit(1)

    NUM_FEATURES = len(selected_features)
    
    contact_ai = ContactAI(num_features=NUM_FEATURES)
    if contact_ai.detection_model is None or contact_ai.localization_model is None:
        logging.error("Model loading failed. Exiting application.")
        sys.exit(1)

    data_logger = None
    save_data = input("Do you want to save the collected data to a CSV file? (y/n): ").lower()
    if save_data == 'y':
        log_headers = ['label','raw_pred', 'smoothed_pred', 'loc_pred'] + selected_features
        data_logger = DataLogger(headers=log_headers)

    default_behaviors = config_loader.load('src/config/default_behaviors.json')
    if not default_behaviors:
        logging.warning("Could not load default behaviors. Continuing with no defaults.")
        default_behaviors = {}

    controller = TaskInterpreter(
        robot=my_robot, 
        ai_model=contact_ai, 
        default_contact_actions=default_behaviors,
        data_logger=data_logger,
        loop_delay=loop_delay,
        realtime_plot=True
    )
    
    controller.load_task_from_file(task_name)
    controller.run()
    
    logging.info("--- System Shutdown ---")
    sys.exit(0)
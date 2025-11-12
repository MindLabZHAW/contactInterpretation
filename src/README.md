# 🤖 Robot Task Interpreter with Real-Time Contact Handling

This project is a Python framework for controlling collaborative robots to perform tasks defined in external JSON files. Its core feature is a real-time monitoring system that uses an AI model to intelligently detect and react to physical contact events, enabling robust and flexible automation.

The system is fully configuration-driven, so you can define complex robot workflows, AI models, and contact-handling behaviors without modifying the Python source code.

## 📖 Table of Contents

  * [Core Architecture](https://www.google.com/search?q=%23core-architecture)
  * [Project Structure](https://www.google.com/search?q=%23project-structure)
  * [Getting Started](https://www.google.com/search?q=%23getting-started)
  * [Prerequisites](https://www.google.com/search?q=%23prerequisites)
  * [Running the Application](https://www.google.com/search?q=%23running-the-application)
  * [Configuration](https://www.google.com/search?q=%23configuration)
  * [Extending the System](https://www.google.com/search?q=%23extending-the-system)
  * [How to Add a New Robot](https://www.google.com/search?q=%23how-to-add-a-new-robot)

## 1\. Core Architecture

The application is built on a modular architecture using two key software design patterns:

  * **Interpreter Pattern**: `task_interpreter.py` reads a sequence of commands from a JSON task file and executes them.
  * **Strategy Pattern**: Supports different robot types via a common `RobotInterface`.

Benefits:

  * ✅ **Flexible**: Add new tasks with JSON files.
  * ✅ **Extensible**: Support new robots via a class following `RobotInterface`.
  * ✅ **Robust**: Define default or step-specific reactions to unexpected contact events.
  * ✅ **Configurable AI**: Swap AI models (e.g., `TransformerModel`, `cnnBiLSTM`, `DMLClassificationNet`) just by changing a line in a YAML file.

## 2\. Project Structure

```
contactInterpretation/
├── pipelines_transferLearning
│   ├── /config/  #AI Model configurations (from training)
│       ├── baselineModels/
│       │   └── baselineFrankaMain.yaml
│       ├── bestModels/
│       │   └── _Transformer1FrankaMainBest.yaml
│       └── fineTunedModels/
│           └── fineTuningCNNBiLSTM1FrankaMainTOUR5.yaml
│
└── src/
    ├── main.py                  # Main entry point for the application
    ├── config_loader.py         # Utility for loading YAML and JSON configs
    ├── data_logger.py           # Handles logging of session data to CSV
    │
    ├── contact_interpretation/
    │   ├── interpreter.py       # Manages the AI models (now config-driven)
    │   └── plotting.py          # Real-time and post-session plotting
    │
    ├── robot_control/
    │   ├── robot_interface.py   # Abstract base class for all robots
    │   ├── robots.py            # Concrete implementations for Franka, UR, etc.
    │   ├── task_interpreter.py  # Core logic for executing tasks
    │   └── robot_teaching.py    # Interactive interface for teaching new tasks
    │
    └── config/                  # Deployment configurations
        ├── config.yaml              # Main profile config
        ├── default_behaviors.yaml   # Default actions for contact
        └── robot_tasks/             # Directory for all task sequence files
            ├── FrankaMain_multi_pose_task.json
            └── my_simulation_task.json
```

## 3\. Getting Started

### Prerequisites

  * Python 3.10+
  * pip (Python package installer)
  * Robot-specific libraries: `ur-rtde`, `franky-control`
  * Config file parser: `pyyaml`

Install required libraries:

```bash
pip install pyyaml ur-rtde franky-control
```

### ⚠️ A Note on Franka Versioning (`franky-control`)

The `franky-control` library must be compatible with your robot's firmware version. You may see this error:

`Incompatible library version (server version: 5, library version: 9)`

This means your robot's firmware (server) is on version 5, but `pip` installed a library for version 9.

To fix this, you must install the `franky-control` wheel that is bundled with the correct `libfranka` version (e.g., `0.9.2` for server v5).

```bash
# Uninstall the wrong version
pip uninstall franky-control

# Install the correct version (example for libfranka 0.9.2)
VERSION=0-9-2
wget https://github.com/TimSchneider42/franky/releases/latest/download/libfranka_${VERSION}_wheels.zip
unzip libfranka_${VERSION}_wheels.zip
pip install numpy
pip install --no-index --find-links=./dist franky-control
```

### Running the Application

Run the application as a module from the **project root (`contactInterpretation/`)**:

```bash
python -m src.main
```

> ⚠️ Note: Running `python src/main.py` directly will cause `ImportError` or prevent configs from loading correctly.

## 4\. Configuration

The system is controlled by three main types of configuration files:

### 1\. `src/config/config.yaml` (Main Profile)

This is the main entry point. It defines "profiles" that bundle a robot, a task, and an AI model together. You only need to change the `active_robot_profile` key to switch the entire application's behavior.

```yaml
robot_profiles:
  
  franka_main_best_model:
    robot_class: "FrankaRobot"
    robot_init_args:
      ip_address: "10.10.10.150"
      robot_name: "franka_main"
      selected_features: ["e0", "e1", "e2", "e3", "e4", "e5", "e6"]
    task_file: "src/config/robot_tasks/FrankaMain_multi_pose_task.json"
    ai_model_config: "config/bestModels/_Transformer1FrankaMainBest.yaml"
    loop_delay: 0.005

  simulation_dml:
    robot_class: "SimulationRobot"
    robot_init_args:
      csv_file_path: "logs/my_log_file.csv"
      selected_features: ["e0", "e1", "e2", "e3", "e4", "e5", "e6"]
    task_file: "src/config/robot_tasks/my_simulation_task.json"
    ai_model_config: "config/baselineModels/baselineFrankaMain.yaml"
    loop_delay: 0.0

# Select which profile to run
active_robot_profile: "simulation_dml"
```

### 2\. `config/bestModels/my_model.yaml` (AI Model Config)

These are the YAML files from your training pipeline. `main.py` loads one of these based on the `ai_model_config` path in the active profile. The `interpreter.py` class then parses this file to load the correct model architecture and weights.

### 3\. `src/config/robot_tasks/my_task.json` (Task File)

This JSON file defines the sequence of actions for the robot. The `task_interpreter.py` reads this file and executes the steps one by one. You can also define step-specific contact reactions here.

```json
{
  "task_name": "My Assembly Task",
  "steps": [
    {
      "step_id": 1,
      "description": "Move to part",
      "command": "move",
      "joints_positions": [ -0.1, -0.1, -0.1, -2.6, -0.0, 2.5, 0.8 ],
      "on_contact": {
        "action": "stop_task",
        "message": "Crashed while moving to part!"
      }
    },
    {
      "step_id": 2,
      "description": "Wait for 5 seconds",
      "command": "wait",
      "duration": 5.0
    }
  ]
}
```

### 4\. `src/config/default_behaviors.yaml`

This YAML file defines fallback actions if a contact occurs on a step that does *not* have its own `on_contact` rule. For example, if contact happens during the "wait" step above, the system will look here.

```yaml
wait:
  action: "log_and_continue"
  message: "Default: Contact detected during 'wait'. Logging."
```

## 5\. Extending the System

### How to Add a New Robot

Thanks to the new design, this is much simpler and does **not** require editing `main.py`.

1.  Open `src/robot_control/robots.py`.
2.  Create a new class that inherits from `RobotInterface` (e.g., `class KUKARobot(RobotInterface):`).
3.  Implement all abstract methods: `connect`, `disconnect`, `get_data`, `send_action`, `is_performing_action`, and `stop`.
4.  Open `src/config/config.yaml`.
5.  Add a new profile under `robot_profiles` that specifies your new `robot_class` and its settings (IP address, etc.).

The application will now see your new robot as an available profile.
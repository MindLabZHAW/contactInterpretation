# 🤖 Robot Task Interpreter with Real-Time Contact Handling

<p align="center">
<img alt="Python Version" src="https://img.shields.io/badge/python-3.9%2B-blue.svg">
<img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg">
<img alt="Status" src="https://img.shields.io/badge/status-active-brightgreen.svg">
</p>

This project is a Python framework for controlling collaborative robots to perform tasks defined in external JSON files. Its core feature is a real-time monitoring system that uses an AI model to intelligently detect and react to physical contact events, enabling robust and flexible automation.

The system is fully configuration-driven, so you can define complex robot workflows and contact-handling behaviors without modifying the Python source code.

## 📖 Table of Contents

* [Core Architecture](#core-architecture)
* [Project Structure](#project-structure)
* [Getting Started](#getting-started)
* [Prerequisites](#prerequisites)
* [Running the Application](#running-the-application)
* [Configuration](#configuration)
* [Extending the System](#extending-the-system)
* [How to Add a New Robot](#how-to-add-a-new-robot)

## 1. Core Architecture

The application is built on a modular architecture using two key software design patterns:

* **Interpreter Pattern**: `task_interpreter.py` reads a sequence of commands from a JSON task file and executes them.
* **Strategy Pattern**: Supports different robot types via a common `RobotInterface`.

Benefits:

* ✅ Flexible: Add new tasks with JSON files.
* ✅ Extensible: Support new robots via a class following `RobotInterface`.
* ✅ Robust: Define default or step-specific reactions to unexpected contact events.

## 2. Project Structure

```
contactInterpretation/
├── src/
│   ├── main.py                  # Main entry point for the application
│   ├── task_interpreter.py      # Core logic for executing tasks
│   ├── robot_teaching.py        # Interactive interface for teaching new tasks
│   ├── config_loader.py         # Utility for loading JSON configuration files
│   ├── data_logger.py           # Handles logging of session data to CSV
│   │
│   ├── contact_interpretation/
│   │   ├── interpreter.py       # Manages the AI models and prediction logic
│   │   └── plotting.py          # Real-time and post-session plotting
│   │
│   ├── robot_control/
│   │   ├── robot_interface.py   # Abstract base class for all robots
│   │   └── robots.py            # Concrete implementations for Franka, UR, etc.
│   │
│   └── config/
│       ├── config.json              # Main configuration for robots and settings
│       ├── default_behaviors.json   # Default actions to take upon contact
│       └── robot_tasks/             # Directory for all task sequence files
│           ├── FrankaMain_multi_pose_task.json
│           └── UR5e_multi_pose_task.json
│
└── pipelines/                   # (Assumed) Contains model training code
    └── trained_models/          # (Assumed) Directory for trained .pth model files
```

## 3. Getting Started

### Prerequisites

* Python 3.9+
* pip (Python package installer)
* Robot-specific libraries: `ur-rtde`, `frankx`

Install required libraries:

```bash
pip install ur-rtde frankx
```

### Running the Application

Run the application as a module from the project root:

```bash
cd /path/to/contactInterpretation
python -m src.main
```

> ⚠️ Note: Running `python src/main.py` directly will cause ImportError.

## 4. Configuration

* **my\_assembly\_task.json**: Defines the robot task sequence.
* **default\_behaviors.json**: Defines fallback actions for contact events, ensuring safety if a task step lacks a specific `on_contact` rule.

## 5. Extending the System

### How to Add a New Robot

1. Open `src/robots.py`.
2. Create a new class inheriting from `RobotInterface` (e.g., `class KUKARobot(RobotInterface):`).
3. Implement all abstract methods: `connect`, `disconnect`, `get_data`, `send_move_command`, `is_moving`, `stop`.
4. In `main.py`, import and instantiate your new robot class.

The `TaskInterpreter` will automatically support your new robot without additional changes.

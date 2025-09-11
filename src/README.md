🤖 Robot Task Interpreter with Real-Time Contact Handling
<p align="center">
<img alt="Python Version" src="https://www.google.com/search?q=https://img.shields.io/badge/python-3.9%2B-blue.svg">
<img alt="License" src="https://www.google.com/search?q=https://img.shields.io/badge/license-MIT-green.svg">
<img alt="Status" src="https://www.google.com/search?q=https://img.shields.io/badge/status-active-brightgreen.svg">
</p>

This project is a Python framework for controlling collaborative robots to perform tasks defined in external JSON files. Its core feature is a real-time monitoring system that uses an AI model to intelligently detect and react to physical contact events, enabling robust and flexible automation.

The entire system is configuration-driven, meaning you can define complex robot workflows and contact-handling behaviors without modifying the Python source code.

📖 Table of Contents
Core Architecture

Project Structure

Getting Started

Prerequisites

Running the Application

Configuration

Extending the System

How to Add a New Robot

1. Core Architecture
The application is built on a clean, modular architecture that combines two key software design patterns:

Interpreter Pattern: The central component (task_interpreter.py) acts as an interpreter. It reads a sequence of commands from a JSON task file and executes them one by one.

Strategy Pattern: The system can work with different types of robots ("strategies") without changing the main logic. This is achieved by defining a common RobotInterface that all specific robot classes (e.g., UR5_Robot) must follow.

This design makes the system:

✅ Flexible: Define new robot tasks just by creating a new JSON file.

✅ Extensible: Add support for new robots by creating a new class that follows the RobotInterface contract.

✅ Robust: Define default and step-specific reactions to unexpected contact events.

2. Project Structure
The project is organized into config for user-defined tasks and src for the application's source code.

contactInterpretation/
├── 📁 config/
│   ├── default_behaviors.json   # Defines default reactions to contact.
│   └── my_assembly_task.json    # Defines a specific robot task.
│
├── 📁 src/
│   ├── __init__.py
│   ├── config_loader.py         # Utility to load JSON files.
│   ├── main.py                  # Main entry point.
│   ├── robot_interface.py       # The abstract "contract" for all robots.
│   ├── robots.py                # Concrete robot implementations (UR5, Franka).
│   └── task_interpreter.py      # The core logic for executing tasks.
│
└── .gitignore                     # (Recommended) To ignore __pycache__ etc.

3. Getting Started
Prerequisites
Python 3.9+

Pip (Python package installer)

Robot-specific libraries (e.g., ur-rtde, frankx)

Install the required libraries:

pip install ur-rtde frankx

Running the Application
The application must be run as a Python module from the project's root directory to ensure all imports are resolved correctly.

Navigate to the root directory in your terminal:

cd /path/to/contactInterpretation

Run the application using the -m flag:

python -m src.main

Note: Running python src/main.py directly from the root will cause an ImportError.

4. Configuration
The application's behavior is controlled entirely by the JSON files in the config/ directory.

my_assembly_task.json: This file defines the sequence of operations for the robot. You can create multiple task files for different jobs.

default_behaviors.json: This file defines fallback actions for contact events based on the command type (e.g., any move should stop the task by default). This provides a safety net if a step in the task file doesn't have its own specific on_contact rule.

5. Extending the System
How to Add a New Robot
The architecture makes it simple to add support for new robots.

Open src/robots.py.

Create a new class that inherits from RobotInterface (e.g., class KUKARobot(RobotInterface):).

Implement all the abstract methods defined in robot_interface.py (connect, disconnect, get_data, send_move_command, is_moving, stop) using the specific Python library for the new robot.

In main.py, import and instantiate your new robot class instead of the existing ones.

The TaskInterpreter will now work with your new robot without any further changes.
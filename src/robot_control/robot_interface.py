from abc import ABC, abstractmethod

class RobotInterface(ABC):
    """
    Abstract base class defining the contract for all robot implementations.
    This ensures that the TaskInterpreter can work with any supported robot.
    """
    @abstractmethod
    def connect(self) -> bool:
        """Establishes a connection to the robot."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Closes the connection to the robot."""
        pass

    @abstractmethod
    def get_data(self) -> dict:
        """Retrieves the current state data from the robot (e.g., joint errors)."""
        pass

    @abstractmethod
    def send_action(self, action_data: dict) -> None:
        """Sends a command or action to the robot (e.g., move, screw)."""
        pass

    @abstractmethod
    def is_performing_action(self) -> bool:
        """Returns True if the robot is currently busy executing an action."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Immediately halts any ongoing robot motion."""
        pass
import json

class ConfigLoader:
    """
    A utility class responsible for loading and parsing data from any
    JSON file. It includes error handling for missing files or bad syntax.
    """
    def load(self, file_path: str) -> dict:
        """
        Loads data from a specified JSON file.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            dict: A dictionary containing the parsed data, or None if an error occurs.
        """
        print(f"Loading JSON from '{file_path}'...")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            print(f"✅ Successfully loaded '{file_path}'.")
            return data
        except FileNotFoundError:
            print(f"ERROR: File not found at path: {file_path}")
            return None
        except json.JSONDecodeError:
            print(f"ERROR: Could not parse the JSON file. Check for syntax errors in {file_path}")
            return None
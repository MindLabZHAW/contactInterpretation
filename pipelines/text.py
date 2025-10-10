def remove_warning_lines(input_file_path, output_file_path):
    """
    Reads an input file, removes lines containing 'WARNING',
    and saves the result to an output file.

    Args:
        input_file_path (str): The path to the source text file.
        output_file_path (str): The path to the destination text file.
    """
    try:
        with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
            for line in infile:
                # Check if 'WARNING' (case-sensitive) is not in the line
                if 'WARNING' not in line:
                    outfile.write(line)
        print(f"Successfully processed the file. Lines with 'WARNING' were removed.")
        print(f"Clean content saved to: {output_file_path}")
    except FileNotFoundError:
        print(f"Error: The file '{input_file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Example Usage ---
if __name__ == "__main__":
    # Define the input and output file names
    input_filename = "/home/rzma/myProjects/contactInterpretation/pipelines/trained_models/franka_main/contact_detection_transformer/64/training_log_hyperparam_search_20251006-151637.txt"
    output_filename = "cleaned_log_training_log_hyperparam_search_20251006-151637.txt"

    # Call the function to perform the operation
    remove_warning_lines(input_filename, output_filename)

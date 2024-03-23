import os
import json
import csv
import argparse

def convert_json_to_csv(input_folder, output_folder):
    # List all JSON files in the folder
    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]

    print(json_files)

    # Define the fields you want to extract
    fields_to_extract = ["ID", "ID_original_dataset", "duration", "duration_no_answer", "wav", "wav_no_answer", "spk_id", "class"]

    # Loop through each JSON file
    for json_file in json_files:
        # Construct the full path to the JSON file
        json_file_path = os.path.join(input_folder, json_file)

        # Load and parse the JSON data
        with open(json_file_path, 'r') as file:
            json_data_list = json.load(file)

        # Create a CSV file for each JSON file
        csv_file_path = os.path.join(output_folder, f"{json_file.replace('.json', '.csv')}")

        # Open the CSV file in write mode
        with open(csv_file_path, 'w', newline='') as csv_file:
            # Create a CSV writer object
            csv_writer = csv.DictWriter(csv_file, fieldnames=fields_to_extract)

            # Write the header to the CSV file
            csv_writer.writeheader()

            # Extract the desired fields and write to CSV
            for row in json_data_list:
                row_data = {field: row[field] for field in fields_to_extract}
                csv_writer.writerow(row_data)

        print(f"CSV file '{csv_file_path}' has been created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert JSON files to CSV format.')
    parser.add_argument('--input_folder', type=str, help='Path to the folder containing JSON files.')
    parser.add_argument('--output_folder', type=str, help='Path to the folder where CSV files will be saved.')

    args = parser.parse_args()

    if not args.input_folder or not args.output_folder:
        parser.error('Please provide both input and output folder paths.')

    os.makedirs(args.output_folder, exist_ok=True)
    convert_json_to_csv(args.input_folder, args.output_folder)

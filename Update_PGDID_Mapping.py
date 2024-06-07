import pandas as pd
import json
import re

"""Once you have downloaded the pgid_mapping.json and Drug_Synonym_Library.csv files, You may use this if you decide to edit or personalize the library file"""

def update_pgid_file(synonym_files, pgid_file='pgid_mapping.json'):
    # Ensure synonym_files is a list
    if isinstance(synonym_files, str):
        synonym_files = [synonym_files]

    try:
        # Attempt to load existing PGDID mapping from the file
        with open(pgid_file, 'r') as file:
            pgid_mapping = json.load(file)
    except FileNotFoundError:
        print("PGDID mapping file not found. Creating a new one.")
        pgid_mapping = {}

    current_pgid = len(pgid_mapping) + 1  # Start numbering from the last ID + 1

    for file_name in synonym_files:
        try:
            # Read the CSV file in chunks
            with pd.read_csv(file_name, chunksize=1000, encoding='latin1') as reader:
                for chunk in reader:
                    chunk['Drug'] = chunk['Drug'].str.lower()  # Convert drug names to lower case
                    for index, row in chunk.iterrows():
                        common_name = row['Drug'].strip()

                        # Skip if common_name is not a string or is empty
                        if not isinstance(common_name, str) or not common_name:
                            continue

                        # Assign PGDID to the common name if not already assigned
                        if common_name not in pgid_mapping:
                            pgid_mapping[common_name] = f'PGDID{current_pgid:05d}'
                            current_pgid += 1

                        # Handle synonyms, ensuring complex entries are parsed correctly
                        synonyms_text = row['Synonyms']

                        # Skip if synonyms_text is not a string
                        if not isinstance(synonyms_text, str):
                            continue

                        raw_synonyms = re.findall(r"'([^']*)'|\b([^,]+)", synonyms_text)
                        synonyms = [syn.strip().lower() for syn_group in raw_synonyms for syn in syn_group if syn]

                        # Assign PGDID to each synonym
                        for synonym in synonyms:
                            if synonym not in pgid_mapping:
                                pgid_mapping[synonym] = f'PGDID{current_pgid:05d}'
                                current_pgid += 1
        except Exception as e:
            print(f"Failed to read/process file {file_name}: {e}")

    # Save the updated mapping to the JSON file
    try:
        with open(pgid_file, 'w') as file:
            json.dump(pgid_mapping, file, indent=4)
            print("PGDID mapping saved successfully.")
    except Exception as e:
        print(f"Failed to save PGDID mapping: {e}")

    return pgid_mapping


update_pgid_file('Drug_Synonym_Library.csv')
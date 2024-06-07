import os
import pandas as pd
import re
import requests
import json
import time
from string import capwords
from itertools import islice
from typing import List, Dict, Any, Tuple, Iterator, Union
import concurrent.futures


# List of common English words to avoid false positives
common_words = {
    'a', 'an', 'and', 'are', 'as', 'at', 'all', 'once', 'be', 'but', 'by', 'for', 'if', 'in', 'is', 'it', 'of', 'on', 'or',
    'the', 'to', 'with', 'his', 'he', 'she', 'her', 'they', 'them', 'cholesterol',
    'time', 'year', 'people', 'way', 'day', 'man', 'thing', 'woman', 'life', 'child', 'world', 'school', 'state',
    'family', 'student', 'group', 'country', 'problem', 'hand', 'part', 'place', 'case', 'week', 'company', 'system',
    'program', 'question', 'work', 'government', 'number', 'night', 'point', 'home', 'water', 'room', 'mother', 'area',
    'money', 'story', 'fact', 'month', 'lot', 'right', 'study', 'book', 'eye', 'job', 'word', 'business', 'issue',
    'side', 'kind', 'head', 'house', 'service', 'friend', 'father', 'power', 'hour', 'game', 'line', 'end', 'member',
    'law', 'car', 'city', 'community', 'name', 'president', 'team', 'minute', 'idea', 'kid', 'body', 'information',
    'back', 'parent', 'face', 'others', 'level', 'office', 'door', 'health', 'person', 'art', 'war', 'history', 'party',
    'result', 'change', 'morning', 'reason', 'research', 'girl', 'guy', 'moment', 'air', 'teacher', 'force', 'education',
    'reflux', 'arthritis', 'control', 'controlling', 'for', 'relief', 'nose', 'dollar general', 'condition', 'treatment', 'him', 'there',
    'drug', 'drugs', 'you', 'use', 'should', 'same', 'can', 'blood', 'levels', 'cause', 'serotonin syndrome', 'serotonin'
}

def file_exists(file_path: str) -> bool:

    return os.path.isfile(file_path)


def download_files(url: str, destination: str) -> None:

    """
    Download the Drug_Synonym_Library.csv and pgid_mapping.json files from the AWS S3 Bucket
    :param destination: To user repository
    """
    if not file_exists(destination):
        response = requests.get(url)
        response.raise_for_status()  # Corrected this line
        with open(destination, 'wb') as file:
            file.write(response.content)
        print(f"Downloading {destination}")
    else:
        print(f"{destination} already exists, skipping download")

def load_pgid_mapping(pgid_file='pgid_mapping.json') -> Dict[str, str]:
    """
    Loads the PGID mapping from the JSON file
    :param pgid_file: directs to pgid_mapping.json
    :return: the mapping dictionary. If it isn't present it returns an error
    """
    try:
        with open(pgid_file, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print("PGID mapping file not found. Please ensure the mapping file exists.")
        return {}


# Load PGIDs & Synonyms at the start
def initialize_synonym_converter(synonym_files: List[str]) -> 'DrugSynonymConverter':
    return DrugSynonymConverter(synonym_files)

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
            cls._instances[cls]._initialized = False
        return cls._instances[cls]

class DrugSynonymConverter:
    _instance = None
    """Singleton class converting the drug synonyms to their common names / PGDID's"""

    def __new__(cls, synonym_files=None):
        """Creates new instance of the converter class if it doesnt exist"""
        if cls._instance is None:
            cls._instance = super(DrugSynonymConverter, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, synonym_files):
        """Initialize the instance with the synonym file"""
        if not self._initialized:
            self.synonym_files = synonym_files
            self.synonym_dict, self.pgid_dict, self.canonical_dict = self.create_synonym_and_id_dicts()
            self._initialized = True

    def reset_instance(cls):
        """
        Resets the singleton instance. The next call to the constructor will create a new instance.
        """
        if cls._instance:
            cls._instance._initialized = False
            cls._instance = None

    def create_synonym_and_id_dicts(self) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, str], Dict[str, str]]:
        """Create & return dictionary for the drug synonyms/PGDIDs/Canonical names
        :return: Tuple containing synonym dictionary, PGDID dictionary, and canonical dictionary
        """
        synonym_dict = {}
        pgid_dict = {}
        canonical_dict = {}

        for file in self.synonym_files:
            try:
                for chunk in pd.read_csv(file, chunksize=10000, encoding='latin1'):
                    chunk['Drug'] = chunk['Drug'].str.lower().str.strip()
                    chunk = chunk.dropna(subset=['Drug'])

                    for index, row in chunk.iterrows():
                        common_name = row['Drug']
                        synonyms_text = row['Synonyms']
                        try:
                            raw_synonyms = re.findall(r"'([^']*)'|\b([^,]+)", synonyms_text)
                            synonyms = [syn.strip().lower() for syn_group in raw_synonyms for syn in syn_group if syn]

                            pgid = pgid_mapping.get(common_name, None)
                            if pgid:
                                canonical_dict[pgid] = common_name.capitalize()
                                synonym_dict[common_name] = (common_name, pgid)
                                pgid_dict[pgid] = common_name

                            for syn in synonyms:
                                pgid = pgid_mapping.get(syn, None)
                                if pgid:
                                    synonym_dict[syn] = (common_name, pgid)
                                    pgid_dict[pgid] = common_name
                            synonym_dict[common_name] = (common_name, pgid)
                        except TypeError:
                            pass

            except pd.errors.EmptyDataError:
                print(f"No data found in the file: {file}")

        self.synonym_dict = synonym_dict
        self.pgid_dict = pgid_dict
        self.canonical_dict = canonical_dict
        return synonym_dict, pgid_dict, canonical_dict

    def convert_to_common_name(self, drug_name: str) -> Tuple[str, str]:
        """
        Convert drug name to common name and PGDID
        :param drug_name: drug name to convert
        :return: tuple--common name, PGDID, or None if no exact match
        """
        drug_name_lower = drug_name.lower().strip()
        if len(drug_name_lower) == 1 or drug_name_lower in common_words:
            return None, None
        if drug_name_lower in self.synonym_dict:
            common_name, pgid = self.synonym_dict[drug_name_lower]
            common_name_capitalized = capwords(common_name)
            return common_name_capitalized, pgid
        return None, None

    def get_pgid_for_synonym(self, drug_synonym: str) -> str:
        """return PGDID for the synonym"""
        return self.pgid_dict.get(drug_synonym.lower())

    def get_drug_names_for_pgids(self, pgids: List[str]) -> Iterator[str]:
        """return drug name for a list of PGDID"""
        return (self.pgid_dict.get(pgid, "Unknown PGDID") for pgid in pgids)

    def get_canonical_for_pgid(self, pgid: str) -> str:
        """Return the canonical name for given PGDID"""
        return self.canonical_dict.get(pgid)

    def get_synonyms_for_drug_name(self, drug_name: str, max_synonyms: int) -> List[str]:
        """
        Return the list of synonyms for a given drug name
        :param drug_name: drug name being looked up
        :param max_synonyms: max number of synonyms to return
        :return: the list of synonyms
        """
        common_name, pgid = self.convert_to_common_name(drug_name)
        if common_name:
            synonyms = (syn for syn, (_, sid) in self.synonym_dict.items() if sid == pgid and syn != common_name)
            return list(islice(synonyms, max_synonyms))
        return []


def tokenize(text: str) -> List[str]:
    """
    Tokenize the input text into words.
    """
    return re.findall(r'\b\w+\b', text.lower())


def find_drugs_in_text(text: str, converter: DrugSynonymConverter) -> List[Tuple[str, str]]:
    """
    Find drugs in the input text by searching for exact matches in the drug synonym library.
    """
    tokens = tokenize(text)
    drugs = []
    for i in range(len(tokens)):
        for j in range(i, len(tokens)):
            phrase = ' '.join(tokens[i:j+1])
            if len(phrase) == 1 or phrase in common_words:
                continue
            if phrase in converter.synonym_dict:
                common_name, pgid = converter.synonym_dict[phrase]
                drugs.append((common_name, pgid))
    return drugs


def validate_input(drug_ids: Union[str, int, List[Union[str, int]]]) -> Tuple[List[Union[str, int]], str]:
    """Validate the input drug PGDID's
        :param drug_ids: PGDID's to validate
        "return: tuple containing the list of PGDIDs and an error message if needed"""
    if not isinstance(drug_ids, list):
        drug_ids = [drug_ids]

    if not all(isinstance(drug_id, (str, int)) for drug_id in drug_ids):
        return [], "The 'drug_ids' parameter must be a list of drug names, PGIDs, or identifiers"

    if not drug_ids:
        return [], "The 'drug_ids' parameter must not be empty"

    return drug_ids, None


def pg_lookup(drug_ids: Union[str, int, List[Union[str, int]]],
              search_by: str = 'pgid',
              fetch: List[str] = None,
              max_synonyms: int = 10) -> List[Dict[str, Any]]:
    """
    Retrieves drug synonym information based on user-defined drug IDs or PGDIDs.

    Parameters:
    - drug_ids (list): List of drug names, drug IDs from other databases, or any other relevant identifiers based on the user preferences.
    - search_by (str): Specifies whether to search by 'drug_name', 'pgid', or 'synonym'.
    - fetch (list): List of fields to fetch from the results.
        -['searched_name', 'common_name', 'pgid', 'synonyms']
    - max_synonyms (int): Maximum number of synonyms to retrieve.

    Returns:
    - A list of dictionaries containing the specified information for each drug.
    """

    if not isinstance(drug_ids, list):
        drug_ids = [drug_ids]

    if not all(isinstance(drug_id, (str, int)) for drug_id in drug_ids):
        raise ValueError("The 'drug_ids' parameter must be a list of drug names, PGIDs, or identifiers")

    if search_by not in ['pgid', 'synonym', 'drug_name']:
        raise ValueError("Invalid search_by parameter. Use 'pgid', 'synonym', or 'drug_name'.")

    if not drug_ids:
        raise ValueError("The 'drug_ids' parameter must not be empty")

    results = []
    for drug_id in drug_ids:
        result = None
        if search_by == 'pgid':
            drug_name = reverse_pgid_mapping.get(drug_id)
            common_name = converter.get_drug_names_for_pgids([drug_id]).__next__()
            if common_name != "Unknown PGDID":
                synonyms = converter.get_synonyms_for_drug_name(common_name, max_synonyms)
                result = {
                    'searched_pgid': drug_id,
                    'searched_name': drug_name,
                    'common_name': common_name,
                    'synonyms': synonyms
                }
            else:
                result = {
                    'searched_pgid': drug_id,
                    'searched_name': "Unknown PGDID",
                    'common_name': "Unknown PGDID",
                    'synonyms': []
                }
        elif search_by in ['drug_name', 'synonym']:
            drug_name, pgid = converter.convert_to_common_name(drug_id)
            if drug_name:
                synonyms = converter.get_synonyms_for_drug_name(drug_name, max_synonyms)
                result = {
                    'searched_name': drug_id,
                    'common_name': drug_name,
                    'pgid': pgid,
                    'synonyms': synonyms
                }
        if result:
            results.append(result)

    if fetch:
        results = [{field: result[field] for field in fetch if field in result} for result in results]

    return results


class DrugRecognition:
    """
    Class for recognizing drugs in bodies of text using the DrugSynonymConverter Class
    """

    def __init__(self, synonym_files: List[str]):
        self.converter = DrugSynonymConverter(synonym_files)

    def process_text(self, text: str) -> List[Tuple[str, str]]:
        return find_drugs_in_text(text, self.converter)

    @staticmethod
    def process_documents(documents: List[str], drug_recognizer, max_workers: int = 4) -> List[List[Tuple[str, str]]]:
        """
        Process mutiple doc's or texts to find the drugs within using multithreading

        :param documents: List of input documents
        :param drug_recognizer: instances for DrugRecognition
        :param max_workers: Total number of worker threads
        :return: List of lists of tuples containing common names and PGDID's of the drugs extracted
        """
        start_time = time.time()  # Start the timer

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(drug_recognizer.process_text, documents))

        end_time = time.time()  # Stop the timer
        processing_time = end_time - start_time  # Calculate the processing time

        print(f"Processing time: {processing_time:.4f} seconds")  # Print the processing time
        return results


def recognize_drugs_in_text(text: str) -> List[Tuple[str, str]]:
    return drug_recognizer.process_text(text)

def recognize_drugs_in_documents(documents: List[str], max_workers: int = 4) -> List[List[Tuple[str, str]]]:
    return DrugRecognition.process_documents(documents, drug_recognizer, max_workers)

def initialize():
    global converter, drug_recognizer, pgid_mapping, reverse_pgid_mapping

    '''URL's for S3 & Download'''
    library_url = 'https://drsyn.s3.amazonaws.com/Drug_Synonym_Library.csv'
    mapping_url = 'https://drsyn.s3.amazonaws.com/pgid_mapping.json'
    download_files(library_url, 'Drug_Synonym_Library.csv')
    download_files(mapping_url, 'pgid_mapping.json')

    synonym_files = ['Drug_Synonym_Library.csv']
    pgid_mapping = load_pgid_mapping()
    reverse_pgid_mapping = {v: k for k, v in pgid_mapping.items()}  # Reverse mapping

    converter = initialize_synonym_converter(synonym_files)
    drug_recognizer = DrugRecognition(synonym_files)

# Call the initialization function when the module is imported
initialize()

# if __name__ == "__main__":

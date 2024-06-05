import pandas as pd
import re
import json
from string import capwords
from itertools import islice
import spacy
import medspacy
from medspacy.ner import TargetRule
from medspacy.context import ConText
from medspacy.section_detection import Sectionizer
from medspacy.visualization import visualize_ent, visualize_dep


def load_pgid_mapping(pgid_file='pgid_mapping.json'):
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

# Load PGIDs at the start
pgid_mapping = load_pgid_mapping()
reverse_pgid_mapping = {v: k for k, v in pgid_mapping.items()}  # Reverse mapping

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

    def create_synonym_and_id_dicts(self):
        """Create & return dictionary for the drug synonyms/PGDIDs/Canonical names"""
        synonym_dict = {}
        pgid_dict = {}
        canonical_dict = {}

        for file in self.synonym_files:
            try:
                for chunk in pd.read_csv(file, chunksize=1000, encoding='latin1'):
                    chunk['Drug'] = chunk['Drug'].str.lower()
                    chunk = chunk.dropna(subset=['Drug'])

                    for index, row in chunk.iterrows():
                        common_name = row['Drug'].strip()
                        synonyms_text = row['Synonyms']
                        raw_synonyms = re.findall(r"'([^']*)'|\b([^,]+)", synonyms_text)
                        synonyms = [syn.strip().lower() for syn_group in raw_synonyms for syn in syn_group if syn]

                        # Map both the common name and each synonym to the PGID
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

            except pd.errors.EmptyDataError:
                print(f"No data found in the file: {file}")

        self.synonym_dict = synonym_dict
        self.pgid_dict = pgid_dict
        self.canonical_dict = canonical_dict
        return synonym_dict, pgid_dict, canonical_dict

    def convert_to_common_name(self, drug_name):
        """
        Convert drug name to common name and PGDID
        :param drug_name: drug name to convert
        :return: tuple--common name, PGDID, or None if no exact match
        """
        drug_name_lower = drug_name.lower().strip()
        if drug_name_lower in self.synonym_dict:
            common_name, pgid = self.synonym_dict[drug_name_lower]  # Correctly unpack tuple
            common_name_capitalized = capwords(common_name)
            return common_name_capitalized, pgid
        return None, None  # Explicitly return None if no exact match is found

    def get_pgid_for_synonym(self, drug_synonym):
        """return PGDID forthe synonym"""
        return self.pgid_dict.get(drug_synonym.lower())

    def get_drug_names_for_pgids(self, pgids):
        """return drug name for a list of PGDID"""
        return (self.pgid_dict.get(pgid, "Unknown PGDID") for pgid in pgids)

    def get_canonical_for_pgid(self, pgid):
        """Return the canonical name for given PGDID"""
        return self.canonical_dict.get(pgid)

    def get_synonyms_for_drug_name(self, drug_name, max_synonyms):
        """
        Return the list of synonyms for a given drug name
        :param drug_name: drug name
        :param max_synonyms: max number of synonyms to return
        :return: the list of synonyms
        """
        common_name, pgid = self.convert_to_common_name(drug_name)
        if common_name:
            synonyms = (syn for syn, (_, sid) in self.synonym_dict.items() if sid == pgid and syn != common_name)
            return list(islice(synonyms, max_synonyms))
        return []


def pg_lookup(drug_ids, search_by='pgid', fetch=None, max_synonyms=10):
    """
    Retrieves drug synonym information based on user-defined drug IDs or PGIDs.

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
            if common_name != "Unknown PGID":
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
                    'searched_name': "Unknown PGID",
                    'common_name': "Unknown PGID",
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
    def __init__(self, synonym_files):
        self.converter = DrugSynonymConverter(synonym_files)
        self.nlp = medspacy.load(medspacy_enable=["medspacy_pyrush"])

        # Add targeted rules for better entity recognition
        target_matcher = self.nlp.add_pipe("medspacy_target_matcher")
        target_rules = [TargetRule(literal="Tylenol", category="DRUG"),
                        TargetRule(literal="Aspirin", category="DRUG"),
                        TargetRule(literal='Effexor', category="DRUG")]
        target_matcher.add(target_rules)

        # Use ConText to handle negation, historical, and other modifiers
        # Add the ConText component to the pipeline by its name
        self.nlp.add_pipe("medspacy_context")

        # Sectionizer can be used if different sections need different handling
        # Add the Sectionizer component to the pipeline by its name
        self.nlp.add_pipe("medspacy_sectionizer")

    def process_text(self, text):
        doc = self.nlp(text)
        recognized_drugs = []

        for ent in doc.ents:
            print(f"Detected entity: {ent.text} (label: {ent.label_})")
            if ent.label_ == "DRUG":
                drug_name = ent.text.lower()
                common_name_capitalized, pgid = self.converter.convert_to_common_name(drug_name)

                if common_name_capitalized:
                    synonyms = self.converter.get_synonyms_for_drug_name(common_name_capitalized, 3)
                    recognized_drugs.append({
                        'name': common_name_capitalized,
                        'synonyms': synonyms,
                        'pgid': pgid
                    })

        return recognized_drugs


"""

####################### Testing area ######################################

"""


if __name__ == "__main__":

    # load the database
    synonym_files = [r'Drug_Synonym_Library.csv']
    converter = DrugSynonymConverter(synonym_files)

    # If needed, reset the converter to load a new synonym database
    # DrugSynonymConverter.reset_instance(cls=DrugSynonymConverter)

    # User-defined drug IDs or PGIDs
    user_drug_ids = ['demser', 'lidocaine', 'tylenol', 'alcohol', 'Ethanol', 'NSAID', 'zoloft', 'adil']
    user_pg_ids = ['PGDID85008', 'PGDID363095', 'PGDID216745']

    # Specify search criteria and fetch fields
    user_results = pg_lookup(user_drug_ids, search_by='drug_name',
                             fetch=['searched_name', 'common_name', 'pgid', 'synonyms'], max_synonyms=5)

    #PGDID Look up example
    pgid_results = pg_lookup(user_pg_ids, search_by='pgid',
                             fetch=['searched_pgid', 'searched_name', 'common_name',  'synonyms'], max_synonyms=5)

    for result in user_results:
        print(result)
    for result in pgid_results:
        print(result)

    # drug_recognizer = DrugRecognition(synonym_files)
    #
    # # Sample text for testing drug recognition
    # sample_text = "I took some aspirin and Tylenol and Aleve and Pristiq and Ibuprofen for my headache, but later I switched to adderall. Did you ever try to take Viagra? I think its crazy when you could be taking zoloft."
    #
    # # Test drug recognition on the sample text
    # results3 = drug_recognizer.process_text(sample_text)
    # for result3 in results3:
    #     print(result3)
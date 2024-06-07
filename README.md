<div style="position: relative; text-align: center; color: white;">
  <img src="https://github.com/boohag/DrSyn/blob/master/Image_Assets/DR_PGBANNER.png" alt="DrSyn Banner">
</div>

# DrSyn: A Python Package for Drug Name Identification and Standardization
DrSyn is a powerful Python package designed to identify drug names within medical texts, extract them, and convert them to their common names based on MeSH standards. This tool is essential for standardizing drug names across various textual sources, making it invaluable for both healthcare and research purposes.

### Key Features:
- **Robust Drug Name Recognition**: Accurately identifies drug names in medical texts, ensuring comprehensive extraction.
- **MeSH Standard Conversion**: Converts extracted drug names to their common or generic names, adhering to MeSH standards.
- **Extensive Synonym Library**: Utilizes a robust and accurate synonym library with data sourced from PubChem, maintained by NCBI.
- **Versatile Text Processing**: Capable of processing individual sentences or multiple documents to extract and standardize drug names.
- **Enhanced Search Capability**: Includes a `pg_lookup` feature for searching specific drug names or Precision Genetics personal drug identifiers (PGDIDs).

DrSyn ensures consistency and accuracy in drug name identification and standardization, facilitating better communication and data management in healthcare and research.

---

This description highlights the main features and importance of your package while maintaining a professional tone.
## Installation
To install DrSyn, clone the repository and install the required packages:
```bash
git clone https://github.com/boohag/DrSyn.git
cd DrSyn
pip install -r requirements.txt

```
#
#
## Usages
```python
recognize_drugs_in_text(text: str) -> List[Tuple[str, str]]*
```
   - **Purpose**: Demonstrates recognizing drug names within a block of text.
   - **Output**: Identifies and prints each drug mentioned along with its PGID.
#
```python
def recognize_drugs_in_documents(documents: List[str], max_workers: int = 4) -> List[List[Tuple[str, str]]]
```
   - **Purpose**: Demonstrates recognizing drug names within multiple documents concurrently. 'max_workers' can be customized.
   - **Output**: Identifies and prints each drug mentioned in each document along with its PGID.
#
```python
pg_lookup(drug_ids: Union[str, int, List[Union[str, int]]],
              search_by: str = 'pgid',
              fetch: List[str] = None,
              max_synonyms: int = 10) -> List[Dict[str, Any]]
```
   - **Purpose**: Demonstrates looking up drugs by their PGIDs or common names and fetching details such as synonyms.
   - **Output**: Prints detailed information for each drug based on the provided PGIDs or drug names.
#
#
## Data Sources

## Contributing

## Licensing

## Contact

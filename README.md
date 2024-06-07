# DrSyn
## A drug entity recognition tool for medical texs
![alt text](https://github.com/boohag/DrSyn/blob/master/DrSyn_Logo.png)

## Installation

You can install DrSyn using pip:
```python
pip install git+'https://github.com/boohag/DrSyn'
```
## Usages
```python
from drsyn import DrugRecognition

# Initialize the DrugRecognition with the synonym files
synonym_files = ['Drug_Synonym_Library.csv']
drug_recognizer = DrugRecognition(synonym_files)

# Process a sample text
sample_text = "I took some aspirin and Tylenol for my headache."
results = drug_recognizer.process_text(sample_text)

for result in results:
    print(result)

# Process multiple documents
documents = ["I took aspirin.", "Tylenol is good for pain relief."]
results = DrugRecognition.process_documents(documents, drug_recognizer)
for result in results:
    print(result)
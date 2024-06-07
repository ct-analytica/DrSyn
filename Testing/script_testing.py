from DrSyn import DrSyn


def main():
    # Sample text for recognizing drugs in text
    sample_text = (
        "I took some aspirin and Tylenol and Aleve and Pristiq and Ibuprofen for my headache, "
        "but later I switched to Adderall. Did you ever try to take Cymbalta? "
        "I think it's crazy when you could be taking Zoloft."
    )

    # Sample documents for recognizing drugs in multiple documents
    sample_documents = [
        "Aspirin and Ibuprofen are common painkillers.",
        "Tylenol is often used for headaches.",
        "Aleve can be effective for pain relief.",
        "Adderall is used to treat ADHD."
    ]

    # Sample PGDIDs for lookup
    sample_pgids = ['PGDID85008', 'PGDID363095', 'PGDID216745']  # Replace with actual PGDIDs
    sample_drugs = ['Tylenol', 'Advil', 'Immodium']
    # Test recognize_drugs_in_text
    print("Testing recognize_drugs_in_text...")
    recognized_drugs = DrSyn.recognize_drugs_in_text(sample_text)
    print("Recognized drugs in text:")
    for common_name, pgid in recognized_drugs:
        print(f"Drug: {common_name}, PGID: {pgid}")

    # Test recognize_drugs_in_documents
    print("\nTesting recognize_drugs_in_documents...")
    recognized_drugs_in_documents = DrSyn.recognize_drugs_in_documents(sample_documents, max_workers=4)
    print("Recognized drugs in documents:")
    for doc_index, recognized_drugs in enumerate(recognized_drugs_in_documents):
        print(f"Document {doc_index + 1}:")
        for common_name, pgid in recognized_drugs:
            print(f"  Drug: {common_name}, PGID: {pgid}")

    # Test pg_lookup
    print("\nTesting pg_lookup...")
    PGDID_lookup_results = DrSyn.pg_lookup(sample_pgids, search_by='pgid', fetch=['searched_pgid', 'common_name', 'synonyms'])
    drug_lookup_results = DrSyn.pg_lookup(sample_drugs, search_by='drug_name', fetch=['searched_name', 'common_name', 'pgid', 'synonyms'])
    print("PGID and Drug Lookup results:")
    for result in PGDID_lookup_results:
        print(f"PGID: {result['searched_pgid']}, Common Name: {result['common_name']}, Synonyms: {result['synonyms']}")
    for result in drug_lookup_results:
        print(f"Drug: {result['searched_name']}, Common Name: {result['common_name']}, Synonyms: {result['synonyms']}")

if __name__ == "__main__":
    main()
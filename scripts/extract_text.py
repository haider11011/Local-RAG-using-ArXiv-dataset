import json

def extract_text_from_json(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)
    abstracts = [entry["abstract"] for entry in data]
    return abstracts
import os
import json

def save_json_to_directory(data, directory, filename):
    """
    Save a dictionary `data` as a JSON file under the specified `directory` with `filename`.
    """
    os.makedirs(directory, exist_ok=True)  # Create directory if it doesn't exist
    file_path = os.path.join(directory, filename)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"✅ JSON saved to {file_path}")
    return file_path
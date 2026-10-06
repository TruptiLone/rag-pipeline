"""
RAG Pipeline

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_text_file
def load_text_file(path):
    # TODO: read a UTF-8 text file at `path` and return its contents as one string.
    with open(path, "r", encoding="utf-8", newline="") as file:
        return file.read()

# Step 2 - load_text_directory
from pathlib import Path

def load_text_directory(directory):
    # TODO: read every .txt file in `directory` and return their contents as a list of strings
    # Read only .txt files, ordered lexicographically by filename.
    return [
        load_text_file(file)
        for file in sorted(Path(directory).iterdir(), key=lambda file: file.name)
        if file.is_file() and file.suffix == ".txt"
    ]


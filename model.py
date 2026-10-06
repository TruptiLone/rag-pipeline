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

# Step 3 - extract_text_from_html
from html.parser import HTMLParser


def extract_text_from_html(html):
    class TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.parts = []
            self.hidden_depth = 0

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style"):
                self.hidden_depth += 1

        def handle_endtag(self, tag):
            if tag in ("script", "style") and self.hidden_depth:
                self.hidden_depth -= 1

        def handle_data(self, data):
            if not self.hidden_depth:
                self.parts.append(data)

    parser = TextExtractor()
    parser.feed(html)
    parser.close()
    return "".join(parser.parts)

# Step 4 - normalize_text
import unicodedata

def normalize_text(text):
    # TODO: NFKC-normalize the text and collapse runs of whitespace into single spaces.
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.split())


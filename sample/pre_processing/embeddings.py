import csv
import json
from sentence_transformers import SentenceTransformer

from pre_processing.encoder import encode_to_vector

sentences = []
metadata = []

# Load the model
print("Loading model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Pre-processing data...")
# Get the data
with open('apps.csv', encoding='utf-8', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader: 
        sentences.append((
            f"App: {row['App']} | "
            f"Category: {row['Category']} | "
            f"Type: {row['Type']} | "
            f"Content Rating: {row['Content Rating']} | "
            f"Genres: {row['Genres']}"
        ))

        metadata.append({
            "App": row['App'],
            "Category": row['Category'],
            "Type": row['Type'],
            "Content Rating": row['Content Rating'],
            "Genres": row['Genres']
        })

# Create the embeddings
embeddings_float32 = encode_to_vector(sentences, model=model)
embeddings_float32.tofile('embeddings.bin')

# Create metadata
with open('metadata.json', 'w') as file: json.dump(metadata, file)

print("Pre-processing finished.")
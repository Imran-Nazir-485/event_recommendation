import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI
import os
import torch
import ast
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
# Load Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

df=pd.read_excel("events_summary.xlsx")

# df['embedding']=''
# df['embedding'] = df['event_summary'].apply(lambda x: model.encode(x))

def get_embedding(text_emb):
  t=[ast.literal_eval(i) for i in text_emb.split()[1:-1]]
  t=torch.tensor(t)
  return t


import sqlite3
import pandas as pd
import json
import numpy as np

# Connect to SQLite database
conn = sqlite3.connect('embeddings.db')
cursor = conn.cursor()

# Retrieve all records from the database
cursor.execute("SELECT * FROM events")
rows = cursor.fetchall()

# Get column names
columns = [desc[0] for desc in cursor.description]

# Close connection
conn.close()

# Function to convert JSON string back to a list
def convert_json_to_list(json_str):
    try:
        return json.loads(json_str) if json_str else []
    except json.JSONDecodeError:
        return []

# Function to convert binary BLOB back to NumPy array
def convert_blob_to_embedding(blob):
    return np.frombuffer(blob, dtype=np.float32) if blob else None

# Reconstruct DataFrame
df = pd.DataFrame(rows, columns=columns)

# Convert 'tags' column from JSON string back to list
df['tags'] = df['tags'].apply(convert_json_to_list)

# Convert 'embedding' column from BLOB back to NumPy array
df['embedding'] = df['embedding'].apply(convert_blob_to_embedding)

# Display reconstructed DataFrame
# print(df_reconstructed.head())







st.dataframe(df)










# # Apply sentence embedding transformation
# df['embedding_tensor']=df['embedding'].apply(get_embedding)

# t=df['embedding_tensor'][0]

# t=df['embedding'][0]

st.title("Event Recommendation")

input=st.text_area("Enter User Information")
if st.button("Recommend") and input!="":

  #Encode the input text
  input_embedding = model.encode(input).reshape(1, -1)  # Reshape to 2D array for cosine similarity

  # Compute cosine similarity
  df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity([x],input_embedding)[0][0])
  
  # Filter rows with similarity >= 50% (0.5)
  similar_texts=df[df['similarity'] >= 0.60].sort_values(by='similarity', ascending=False)

  # # Display results
  # st.write(similar_texts[['similarity','title', 'location', 'address', 'category','description', 'organizer','tags']])
  st.write(df)



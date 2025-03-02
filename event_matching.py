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
import sqlite3
import json
import numpy as np
import gdown

# Load Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

df=pd.read_excel("events_summary.xlsx")
profile_summary=pd.read_excel("profile_summary_combine.xlsx")
##################################################################################################

def get_embedding(text_emb):
  t=[ast.literal_eval(i) for i in text_emb.split()[1:-1]]
  t=torch.tensor(t)
  return t


# https://drive.google.com/file/d/1NV-lLB8MXRCHH73gHQC7Qyq60qhHxPmN/view?usp=sharing
# Google Drive File ID
file_id = "1NV-lLB8MXRCHH73gHQC7Qyq60qhHxPmN"
output_file = "10_events_embedding.db"

# Download the file
@st.cache_data
def download_db():
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_file, quiet=False)
    return output_file

# Load the database
def load_database():
    db_file = download_db()
    conn = sqlite3.connect(db_file)
    return conn

conn = load_database()
cursor = conn.cursor()

# cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
# tables = cursor.fetchall()
# st.write("Tables in the database:", tables)
# conn.close()


# Connect to SQLite database
# conn = sqlite3.connect('embeddings.db')
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

# st.write(df)

######################################################################################################################################################

# Connect to SQLite database
conn = sqlite3.connect('profiles_db.db')
cursor1 = conn.cursor()

# Retrieve all records from the database
cursor.execute("SELECT * FROM profiles")
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
profile_df = pd.DataFrame(rows, columns=columns)

# Convert 'embedding' column from BLOB back to NumPy array
profile_df['embeddings'] = profile_df['embeddings'].apply(convert_blob_to_embedding)


# st.write(df.shape)

st.title("Event Recommendation")

selection=st.sidebar.selectbox(
    "☰ Menu",
    ["Event Recommendation", "Option 2", "Option 3"])

if selection=="Event Recommendation":
  profile_id=st.selectbox("Select",profile_df["profile_id"])
  embeddings=profile_df[profile_df["profile_id"]==profile_id]['embeddings'].values
  # st.write(embeddings[0])

# input=st.text_area("Enter User Information")
# if st.button("Recommend") and input!="":

#   #Encode the input text
#   input_embedding = model.encode(input).reshape(1, -1)  # Reshape to 2D array for cosine similarity

# #   # Compute cosine similarity
#   # df['embedding'][0]

#   # # Ensure both are numpy arrays
#   # embedding_1 = np.array(df['embedding'][0]).reshape(1, -1)
#   # embedding_2 = np.array(embeddings[0]).reshape(1, -1)

#   # st.write(embedding_1)
#   # st.write(embedding_2)

  
#   # Compute cosine similarity
#   # similarity = cosine_similarity(embedding_1, embedding_2)
  
  df['similarity']=df['embedding'].apply(lambda x: cosine_similarity([x],embeddings[0].reshape(1,-1))[0][0])
  
#Filter rows with similarity >= 50% (0.5)
  similar_texts=df[df['similarity'] >= 0.50].sort_values(by='similarity', ascending=False)
#   st.write(similar_texts.shape)

#Display results
  st.dataframe(similar_texts[['similarity','title', 'location', 'address', 'category','description', 'organizer','tags']])
#   # st.write(df)



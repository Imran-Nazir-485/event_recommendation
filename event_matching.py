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
# model = SentenceTransformer('all-MiniLM-L6-v2')

# df=pd.read_excel("events_summary.xlsx")
# profile_summary=pd.read_excel("profile_summary_combine.xlsx")
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
cursor = conn.cursor()

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
    ["Home" ,"Recommended", "My Profile"])

if selection=="Recommended":
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
  
# #Filter rows with similarity >= 50% (0.5)
  similar_texts=df[df['similarity'] >= 0.50].sort_values(by='similarity', ascending=False)
  similar_texts=similar_texts.drop_duplicates(['title'])

  st.write(similar_texts[:10])
# #   st.write(similar_texts.shape)

# #Display results
#   st.dataframe(similar_texts[['similarity','title', 'location', 'address','price', 'category','link','description','refund_policy', 'organizer','tags']])
#   st.subheader("User Profile", profile_id)
  # st.write(profile_df[profile_df["profile_id"]==profile_id]['profile_summary'].values[0])
#   # st.write(df)


  import streamlit as st
  import random
  
  # Custom CSS for styling
  st.markdown("""
      <style>
      .event-tile {
          background-color: #1E1E1E;
          padding: 15px;
          border-radius: 10px;
          color: white;
          font-family: Arial, sans-serif;
          margin-bottom: 10px;
      }
      .event-title {
          font-size: 20px;
          font-weight: bold;
      }
      .event-details {
          font-size: 16px;
          display: flex;
          align-items: center;
          gap: 8px;
      }
      </style>
  """, unsafe_allow_html=True)
  
  # Generate a random number of events (between 3 and 10)
  num_events = random.randint(1, 10)


  
  # Sample event data
  cities = similar_texts['location'][:10].values
  event_titles = similar_texts['title'][:10].values[0]
  prices = similar_texts['price'][:10].values
  date = similar_texts['date'][:10].values
  
  
  # Loop to generate event tiles dynamically in rows of 2
  for i in range(0, num_events, 2):
      col1, col2 = st.columns(2)  # Create two columns
  
      # Generate first event
      event_name1 = event_titles[i]
      event_date1 = date[i]
      city1 = city[i]
      price1 = prices[i]
  
      event_html1 = f"""
      <div class="event-tile">
          <div class="event-title">{event_name1}</div>
          <div class="event-details">
              📅 {event_date1} - 📍 {city1} - 💰 {price1}€
          </div>
      </div>
      """
  
      col1.markdown(event_html1, unsafe_allow_html=True)
  
      # Generate second event (if available)
      if i + 1 < num_events:
          event_name2 = event_titles[i+1]
          event_date2 = date[i+1]
          city2 = city[i+1]
          price2 = prices[i+1]
  
          event_html2 = f"""
          <div class="event-tile">
              <div class="event-title">{event_name2}</div>
              <div class="event-details">
                  📅 {event_date2} - 📍 {city2} - 💰 {price2}€
              </div>
          </div>
          """
  
          col2.markdown(event_html2, unsafe_allow_html=True)

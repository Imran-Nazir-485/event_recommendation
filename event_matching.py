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
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
# Load Sentence Transformer model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# df=pd.read_excel("events_summary.xlsx")
# profile_summary=pd.read_excel("profile_summary_combine.xlsx")
##################################################################################################

def get_embedding(text_emb):
  t=[ast.literal_eval(i) for i in text_emb.split()[1:-1]]
  t=torch.tensor(t)
  return t


###################################################################################################
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq

llm = ChatGroq(
    temperature=0,
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY
)
############################################################################################################
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
  num_events = random.randint(0, 10)


  
  # Sample event data
  cities = similar_texts['location'][:10].values
  event_titles = similar_texts['title'][:10].values
  prices = similar_texts['price'][:10].values
  date = similar_texts['date'][:10].values
  
  
  # Loop to generate event tiles dynamically
  for i in range(10):
      event_name = event_titles[i]
      event_date = date[i]
      city = cities[i]
      price = prices[i]
  
      event_html = f"""
      <div class="event-tile">
          <div class="event-title">{event_name}</div>
          <div class="event-details">
              📅 {event_date} - 📍 {city} - 💰 {price}€
          </div>
      </div>
      """
  
      st.markdown(event_html, unsafe_allow_html=True)



if selection=="My Profile":
  profile_id=st.selectbox("Select",profile_df["profile_id"])
  my_profile=profile_df[profile_df["profile_id"]==profile_id]['profile_summary'].values[0]
  # st.write(my_profile)

  st.write(llm.invoke(my_profile + "extract key info"))


  # Sample extracted user profile data (replace with real extracted data)
  user_data = {
      "name": "Max Mustermann",
      "profile_picture": "https://via.placeholder.com/100",  # Replace with actual URL
      "location": "Düsseldorf",
      "interests": ["Musik", "Theater", "Kunst", "Fitness"]
  }
  
  # Custom CSS for styling
  st.markdown("""
      <style>
      .profile-card {
          background-color: #1E1E1E;
          padding: 20px;
          border-radius: 12px;
          color: white;
          text-align: center;
          width: 100%;
      }
      .profile-picture {
          border-radius: 50%;
          width: 80px;
          height: 80px;
          margin-bottom: 10px;
      }
      .interest-badge {
          display: inline-block;
          background: #FF5722;
          padding: 5px 10px;
          margin: 5px;
          border-radius: 8px;
          font-size: 14px;
      }
      </style>
  """, unsafe_allow_html=True)
  
  # Profile Card
  st.markdown(f"""
      <div class="profile-card">
          <img src="{user_data['profile_picture']}" class="profile-picture">
          <h2>{user_data['name']}</h2>
          <p>📍 Standort: {user_data['location']}</p>
          <h3>🎯 Interessen</h3>
          <div>
              {''.join([f'<span class="interest-badge">🎵 {interest}</span>' for interest in user_data['interests']])}
          </div>
      </div>
  """, unsafe_allow_html=True)















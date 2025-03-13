import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI
import os
import torch
import ast
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import sqlite3
import json
import numpy as np
import gdown
from dotenv import load_dotenv
import random

from openai import OpenAI
# Load environment variables
load_dotenv()
# Load Sentence Transformer model
# model = SentenceTransformer('all-MiniLM-L6-v2')
# Page Layout


import pickle
import faiss

# Load FAISS index
index = faiss.read_index("faiss_index.bin")

# Load metadata
with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)


file_id = "1ug8pf1M1tes-CJMhS_sso372tvC4RQv8"
output_file = "open_ai_key.txt"

# https://docs.google.com/spreadsheets/d/1Dp6Y9ps4md393F5eRZzaZhu044k4JCmrbYDxWmQ6t2g/edit?gid=0#gid=0
sheet_id = '1Dp6Y9ps4md393F5eRZzaZhu044k4JCmrbYDxWmQ6t2g' # replace with your sheet's ID
url=f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
df=pd.read_csv(url)
os.environ["OPENAI_API_KEY"] = df.keys()[0]


# Query Text
query_text = "Event: Tech Conference | Location: Berlin | Date: 2025-06-10"
# OpenAI API Client
client = OpenAI()
# Convert query to embedding
query_embedding = client.embeddings.create(model="text-embedding-ada-002", input=query_text).data[0].embedding
query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)

# Search in FAISS
D, I = index.search(query_embedding, k=3)  # Get top 3 similar rows

# Display Results
st.write("\n🔍 Top Matching Rows:")
for idx in I[0]:
    st.write(metadata[idx])



# Set the page configuration to wide mode
st.set_page_config(page_title="My App", layout="wide")

# Inject CSS to expand content area
# st.markdown(
#     """
#     <style>
#         /* Increase the width of the main content */
#         .main .block-container {
#             max-width: 98%;
#             padding-left: 2%;
#             padding-right: 2%;
#         }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# Inject custom CSS to adjust sidebar width
st.markdown(
    """
    <style>
        /* Reduce sidebar width */
        [data-testid="stSidebar"] {
            width: 200px !important;
            min-width: 200px !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# df=pd.read_excel("events_summary.xlsx")
# profile_summary=pd.read_excel("profile_summary_combine.xlsx")
##################################################################################################

def get_embedding(text_emb):
  t=[ast.literal_eval(i) for i in text_emb.split()[1:-1]]
  t=torch.tensor(t)
  return t


def get_prompt(profile_data):
  return f"""
  Extract key details from the following user profile data:

  1. **Location**: Extract the city and country (if available).
  2. **Interests**: Extract a list of hobbies or interests.
  3. **Additional Information**: Extract any relevant details such as job title, bio, or age (if available).
  4. No Additional information like "here is the json data"
  
  Format the extracted details as JSON:
  {{
    
    "location": "City, Country",
    "interests": ["Interest 1", "Interest 2", "Interest 3"],
    "additional_info": "Other relevant details"
  }}
  
  # Here is the user data:
  {profile_data}
  """

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
df['price']=df['price'].apply(lambda x: x if re.search(r'\d+', x) else np.nan)
df.dropna(subset=['price'], inplace=True)
df=df.reset_index(drop=True)


to_drop=[]
for i in range(len(df)):
  for j in ['Kein Titel verfügbar','Datum nicht verfügbar','Ort nicht verfügbar','Adresse nicht verfügbar']:
    if j in df.loc[i,'title'] or j in df.loc[i,'date']  or j in df.loc[i,'location'] or j in df.loc[i,'address']:
      to_drop.append(i)
df.drop(to_drop,inplace=True)
df=df.reset_index(drop=True)

# st.write(df)

######################################################################################################################################################

# Connect to SQLite database
conn = sqlite3.connect('user_data_extracted.db')
cursor = conn.cursor()

# Retrieve all records from the database
cursor.execute("SELECT * FROM user_profiles")
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





# Title Section
title_html = """
    <div style="text-align: left; padding: 20px; max-width: 800px; margin: auto;">
        <h1 style="color: #FF914D;">🎉 Event Recommendation App</h1>
    </div>
"""

# About Section

# Display in Streamlit
st.markdown(title_html, unsafe_allow_html=True)

selection=st.sidebar.selectbox(
    "☰ Menu",
    ["Home" ,"Recommended", "My Profile", "Build Profile"])

if selection=="Home":

    
    # Create a slider with a range of 1 to 100
    value = st.slider("Select a number", min_value=50, max_value=df.shape[0], value=50)


    # Generate 100 random numbers
    random_numbers = [random.randint(1, value) for _ in range(df.shape[0])]


    # Sample event data
    cities = df.loc[random_numbers,'location'].values
    event_titles = df.loc[random_numbers,'title'].values
    prices = df.loc[random_numbers,'price'].values
    dates = df.loc[random_numbers,'date'].values
    addresses = df.loc[random_numbers,'address'].values  # Extracting address
    tags_list = df.loc[random_numbers,'tags'].values  # Extracting tags
    
    # Custom CSS for styling event tiles
    st.markdown(
        """
        <style>
        .event-tile {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 10px;
        }
        .event-title {
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }
        .event-details, .event-extra {
            font-size: 14px;
            color: #555;
            margin-top: 5px;
        }
        .tags {
            margin-top: 8px;
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }
        .tag {
            background: #007bff;
            color: white;
            padding: 3px 8px;
            border-radius: 5px;
            font-size: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Loop to generate event tiles dynamically
    for i in range(len(random_numbers)):
        event_name = event_titles[i]
        event_date = dates[i]
        city = cities[i]
        price = prices[i]
        address = addresses[i] if addresses[i] else "No address provided"
        tags = tags_list[i]  # Handling multiple tags
    
        # Generating tags HTML
        tags_html = "".join(f'<span class="tag">{tag.strip()}</span>' for tag in tags)
    
        event_html = f"""
        <div class="event-tile">
            <div class="event-title">{event_name}</div>
            <div class="event-details">
                📅 {event_date} - 📍 {city} - 💰 {price}€
            </div>
            <div class="event-extra">
                🏠 {address}
            </div>
            <div class="event-extra">
                🔖 {tags_html}
            </div>
        </div>
        """
    
        st.markdown(event_html, unsafe_allow_html=True)




    
    # about_html = """
    # <div style="text-align: left; padding: 20px; max-width: 800px; margin: auto;">
    #     <h2 style="color: #33A1FF;">📌 About</h2>
    #     <p style="font-size: 18px; color: #ddd;">
    #         The Event Recommendation App is designed to help users discover the best events in Germany 
    #         based on their interests & location seen in social media profile. Whether you're looking for concerts, sports matches, 
    #         networking meetups, or cultural events, our AI-driven system provides personalized suggestions 
    #         so you never miss out on exciting activities.
    #     </p>
    # </div>
    # """

    # # Key Features Section
    # features_html = """
    #     <div style="text-align: left; padding: 20px; max-width: 800px; margin: auto;">
    #         <h2 style="color: #33A1FF;">✨ Key Features</h2>
    #         <ul style="font-size: 16px; color: #ddd;">
    #             <li>✅ <b>Personalized Event Suggestions:</b> AI-powered recommendations based on your interests.</li>
    #             <li>📍 <b>Location-Based Filtering:</b> Find events happening near you.</li>
    #             <li>🎭 <b>Diverse Event Categories:</b> From music and sports to networking and tech meetups.</li>
    #             <li>📆 <b>Interactive Event Listings:</b> View event details, dates, venues, and ticket availability.</li>
    #             <li>⏳ <b>Real-Time Updates:</b> Stay informed about trending and newly added events.</li>
    #         </ul>
    #     </div>
    # """


    # st.markdown(about_html, unsafe_allow_html=True)
    # st.markdown(features_html, unsafe_allow_html=True)



    
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

  # st.write(similar_texts[:10])
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
  my_profile=profile_df[profile_df["profile_id"]==profile_id]['profile_data'].values[0]
  # st.write(my_profile)
  # user_data=llm.invoke(get_prompt(my_profile)).content
  # json_string = my_profile.replace("'", '"""')
  # st.write(json_string)
  
  # st.write(json.loads(my_profile))
  # st.write(type(user_data))
  user_data=json.loads(my_profile)
  
  # Profile Section
  st.markdown(f"<h1 style='text-align: center;'>Profile ID - {profile_id}</h1>", unsafe_allow_html=True)
  st.markdown(f"<p style='text-align: center;'>📍 <b>Standort:</b> {user_data['location']}</p>", unsafe_allow_html=True)
    
    # Interests Section
  st.markdown("<h3 style='text-align: center;'>🎯 <b>Interessen</b></h3>", unsafe_allow_html=True)
    
  # Define interest colors (repeats if more interests exist)
  colors = ["#FF5733", "#FF914D", "#FF5E78", "#FF7433", "#33A1FF", "#33FF77", "#FFC300", "#A133FF", "#FF3387", "#33FFA5"]
    
    # Custom CSS for styling badges on a white background
  st.markdown(
        """
        <style>
        .badge-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 10px;
            margin-top: 10px;
        }
        .badge {
            background-color: var(--bg-color);
            color: #333; /* Dark text for contrast */
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
            box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1); /* Subtle shadow */
            border: 1px solid rgba(0, 0, 0, 0.2); /* Light border for visibility */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Generate dynamic badges based on interests
  badges_html = '<div class="badge-container">'
  for i, interest in enumerate(user_data["interests"]):
        color = colors[i % len(colors)]  # Cycle through colors if interests exceed color list
        badges_html += f'<span class="badge" style="--bg-color: {color};">{interest}</span>'
  badges_html += '</div>'
    
  st.markdown(badges_html, unsafe_allow_html=True)

  # st.write(profile_df[profile_df["profile_id"]==profile_id]['profile_summary'].values[0])






if selection=="Build Profile":
    import streamlit as st

# # App title
# st.title("🎉 Event Recommendation Form")

    # Sidebar for user info
    st.sidebar.header("User Information")
    name = st.sidebar.text_input("Your Name")
    email = st.sidebar.text_input("Email")

    st.write("## Tell us about your event preferences!")

    # Use a form to collect data
    with st.form("event_preferences_form"):
        # 1. Event Type
        event_types = st.multiselect(
            "What type of events do you enjoy?",
            ["Concerts", "Conferences", "Sports", "Workshops", "Meetups", "Festivals", "Theater", "Networking"]
        )
    
        # 2. Format
        event_format = st.radio("Do you prefer virtual or in-person events?", ["In-Person", "Virtual", "Both"])
    
        # 3. Interests
        interests = st.multiselect(
            "Select your interests",
            ["Music", "Tech", "Business", "Arts", "Gaming", "Food & Drinks", "Networking", "Sports", "Wellness"]
        )
    
        # 4. Time Preferences
        days_available = st.multiselect(
            "What days are you available?",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
        time_preference = st.radio("Preferred event time?", ["Morning", "Afternoon", "Evening", "No Preference"])
    
        # 5. Location
        location = st.text_input("Enter your city or ZIP code")
        travel_distance = st.slider("How far are you willing to travel? (miles)", 1, 100, 10)
    
        # 6. Budget
        budget = st.radio("Event Budget", ["Free", "Under $20", "$20 - $50", "$50 - $100", "No Limit"])
    
        # 7. Social Preferences
        social_preference = st.radio("Do you prefer to attend events alone or with friends?", ["Alone", "With Friends", "Both"])
    
        # 8. Notifications
        notify = st.checkbox("Notify me about new events matching my preferences")
    
        # Submit button
        submitted = st.form_submit_button("Submit")
    
    # Handle form submission
    if submitted:
        user_data = {
            "Name": name,
            "Email": email,
            "Event Types": event_types,
            "Format": event_format,
            "Interests": interests,
            "Availability": days_available,
            "Time Preference": time_preference,
            "Location": location,
            "Travel Distance": travel_distance,
            "Budget": budget,
            "Social Preference": social_preference,
            "Notifications": notify
        }
        
        st.success("✅ Your preferences have been saved!")
        st.json(user_data)  # Show collected data

import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI
import os
import torch
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
# Load Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')


df=pd.read_excel("events_summary.xlsx")


# Apply sentence embedding transformation
df['embedding'] = df['event_summary'].apply(lambda x: model.encode(x))

st.title("Event Matching")

# Print the DataFrame with embeddings
st.write(df)

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

df['embedding']=''
df['embedding'] = df['event_summary'][:100].apply(lambda x: model.encode(x))

def get_embedding(text_emb):
  t=[ast.literal_eval(i) for i in text_emb.split()[1:-1]]
  t=torch.tensor(t)
  return t

# # Apply sentence embedding transformation
# df['embedding_tensor']=df['embedding'].apply(get_embedding)

# t=df['embedding_tensor'][0]

t=df['embedding'][0]

st.title("Event Matching")

input=st.text_input("Enter")
if st.button("Recommend") and input!=None:

  # Encode the input text
  # input_embedding = model.encode(input).reshape(1, -1)  # Reshape to 2D array for cosine similarity

  # Compute cosine similarity
  df['similarity'] = df['embedding'][:100].apply(lambda x: cosine_similarity([x],t.reshape(1,-1))[0][0])
  
  # Filter rows with similarity >= 50% (0.5)
  similar_texts=df[df['similarity'] >= 0.60].sort_values(by='similarity', ascending=False)

  # # Display results
  st.write(similar_texts[['similarity','title', 'location', 'address', 'category','description', 'organizer','tags']])


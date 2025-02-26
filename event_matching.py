import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI
import os
import torch
import ast
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
# Load Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

df=pd.read_excel("events_summary_embeddings.xlsx")


def get_embedding(text_emb):
  t=[ast.literal_eval(i) for i in text_emb.split()[1:-1]]
  t=torch.tensor(t)
  return t

# Apply sentence embedding transformation
df['embedding_tensor']=df['embedding'].apply(get_embedding)

st.title("Event Matching")

# Print the DataFrame with embeddings
st.write(df)

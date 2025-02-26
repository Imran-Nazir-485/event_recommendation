import pandas as pd
import streamlit as st



st.title("Event Matching")


df=pd.read_excel("events_summary.xlsx")
st.write(df)

import pandas as pd
import streamlit as st
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import os
from time import sleep

from requestsAPI import process_tweets, sentiment_detection, sarcasm_detection


# Load the model
net_model = torch.load("saved_models/netModel.pt")

from sidebar import sidebar
countdown = 0
def query_sentiment(query):
    result = sentiment_detection(query)
    print(result)
    if type(result) is not list and result['error']:
        if result['estimated_time']:
            with st.spinner(f'Waiting for Hugging Face: {result["estimated_time"]} seconds'):
                sleep(result['estimated_time']+1)
                #countdown = result['estimated_time']
                
        query_sentiment(query)
    return result

def query_sarcasm(query):
    result = sentiment_detection(query)
    print(result)
    if type(result) is not list and result['error']:
        if result['estimated_time']:
            sleep(result['estimated_time']+1)
        query_sarcasm(query)
    return result


def clear_submit():
    st.session_state["submit"] = False



st.set_page_config(page_title="Disaster Detection", page_icon="📖", layout="wide")
st.header('🐦 Tweet Disaster Detection 🐦')

sidebar()

index = None
modelSelected = True

query = st.text_area("Ask a question about the document", on_change=clear_submit)
with st.expander("Advanced Options"):
    show_all_chunks = st.checkbox("Show all chunks retrieved from vector search")
    show_full_doc = st.checkbox("Show parsed contents of the document")

button = st.button("Submit")
if button or st.session_state.get("submit"):
    if not query:
        st.error("Please enter a prompt!")
    elif not modelSelected:
        st.error("Please select a model!")
    else:
        st.session_state["submit"] = True
        # Output Columns
        answer_col, sources_col = st.columns(2)

        
        sentiment, sarcasm = query_sentiment(query), query_sarcasm(query)

        processed_data = process_tweets(sentiment, sarcasm)
        data = torch.Tensor()

        #data = torch.cat(0, query, processed_data['negative'], processed_data['neutral'],  processed_data['positive'], processed_data['sarcastic'], processed_data['not_sarcastic'])

        answer = net_model.forward(data)
        print(answer)
        



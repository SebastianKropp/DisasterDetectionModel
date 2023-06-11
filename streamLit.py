import pandas as pd
import streamlit as st
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from modelClasses import netmodel
from time import sleep
import math as Math
from requestsAPI import process_tweets, sentiment_detection, sarcasm_detection
from transformers import AutoModelForSequenceClassification




from sidebar import sidebar
countdown = 0
def query_sentiment(query):
    result = sentiment_detection(query)
    print(result)
    if type(result) is not list and result['error']:
        if result['estimated_time']:
            with st.spinner(f'Waiting for Hugging Face: {round(result["estimated_time"])} seconds'):
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
modelSelected = "BertTDD"
input_layers = 3565

query = st.text_area("Enter a Tweet or just Text", on_change=clear_submit)
option = st.selectbox(
    'Which model would you like to use?',
    ('NetTDD', 'BertTDD'))

button = st.button("Submit")
if button or st.session_state.get("submit"):
    if not query:
        st.error("Please enter a prompt!")
    elif not modelSelected:
        st.error("Please select a model!")

    if modelSelected == "BertTDD":
        model = AutoModelForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', num_labels=2)
        model.load_state_dict(torch.load("saved_models/transModel.pt"))
    if modelSelected == "NetTDD":
        model = netmodel(input_layer=input_layers, num_hidden=10, node_per_hidden=input_layers, droppout=0.3)
        model.load_state_dict(torch.load("saved_models/netModel.pt"))

    st.session_state["submit"] = True
    # Output Columns
    answer_col, sources_col = st.columns(2)
        
    sentiment, sarcasm = query_sentiment(query), query_sarcasm(query)


    processed_data = process_tweets(sentiment, sarcasm)

    dummyHot = torch.zeros(3565)
    dummyHot[2] = processed_data['negative']
    dummyHot[3] = processed_data['neutral']
    dummyHot[4] = processed_data['positive']
    dummyHot[5] = processed_data['sarcastic']
    dummyHot[6] = processed_data['not_sarcastic']
        
    model.eval()
    answer = model.forward(dummyHot)
    st.success(f"Answer: {'Yes' if round(float(answer[1]*100000000)) else 'No'}")
    



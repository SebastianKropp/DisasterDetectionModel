import pandas as pd
import streamlit as st
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import os
from time import sleep
import math as Math
from requestsAPI import process_tweets, sentiment_detection, sarcasm_detection
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

from modelClasses import netmodel, preprocess, BERTmodel
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
#st.header('Tweet Disaster Detection')
#st.image('../images/twitter.png', width=100, use_column_width=False)

sidebar()

input_layers = 3565
col1, col2 = st.columns([0.1, 0.9])

with col1:
    st.image(os.getcwd()+'/images/twitterLogo.png', width=100)
    
with col2:
    st.header('Tweet Disaster Detection')
                
query = st.text_area("Enter a Tweet or just Text", on_change=clear_submit)
option = st.selectbox(
    'Which model would you like to use?',
    ('BertTDD', 'NetTDD'))

button = st.button("Submit")
if button or st.session_state.get("submit"):
    if not query:
        st.error("Please enter a prompt!")

    if option == "BertTDD":
        model = AutoModelForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', num_labels=2)
        model.load_state_dict(torch.load(os.getcwd()+"/saved_models/transModel.pt"))
    if option == "NetTDD":
        model = netmodel(input_layer=input_layers, num_hidden=10, node_per_hidden=input_layers, droppout=0.3)
        model.load_state_dict(torch.load(os.getcwd()+"/saved_models/netModel.pt"))

    st.session_state["submit"] = True
    # Output Columns
    answer_col, sources_col = st.columns(2)

    #Cannot use keyword, or location on live data (Twitter API does not allow)
    if option == "BertTDD":
        tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        input_ids, attention_mask = preprocess(query, tokenizer)
        model.eval()
        output = model.forward(input_ids, attention_mask=attention_mask)
        logits = None
        if hasattr(output, 'logits'):
            logits = output.logits
        else:
            logits = output.last_hidden_state[:, 0, :]

        _, predicted_labels = torch.max(logits, dim=1)
        predicted_prob = torch.softmax(logits, dim=1) 
        predicted_prob = predicted_prob.tolist()[0]


        st.success(f"{'The prompt indicates a disaster' if predicted_labels else 'The prompt indicates no potential disasters'}, with a confidence of {round(predicted_prob[predicted_labels] * 100, 2)}%")
    if option == "NetTDD":
        sentiment, sarcasm = False, False

        while not sentiment and not sarcasm:
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
        print(answer)
    
        st.success(f"{'The prompt indicates a disaster' if round(float(answer[1]*100000000)) else 'The prompt indicates no potential disasters'}, with a confidence of a disaster at {round(float(answer[1]*100000000)*100)}%")
    



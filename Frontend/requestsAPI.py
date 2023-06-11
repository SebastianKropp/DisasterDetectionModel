###              EXAMPLE USE:                ###
## tweet_text = "This is a test tweet!"
## sentiment = sentiment_detection(tweet_text)
## sarcasm = sarcasm_detection(tweet_text) 
import requests
API_TOKEN = "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"

def query(API_URL, headers, payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def sentiment_detection(tweet_text):
    # Define the first API endpoint and function
    API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    # Use the first function to query the sentiment of some text
    output_sentiment = query(API_URL, headers, {
        "inputs": tweet_text,
    })

    return output_sentiment


def sarcasm_detection(tweet_text):
    # Define the second API endpoint and function
    API_URL = "https://api-inference.huggingface.co/models/helinivan/english-sarcasm-detector"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
        
    output_sarcasm = query(API_URL, headers, {
        "inputs": tweet_text,
    })

    return output_sarcasm

def process_tweets(sentiment_result, sarcasm_result):
    row = {}

    # Add the sentiment and sarcasm probabilities to the row
    print (sentiment_result, sarcasm_result)
    try:
        row['negative'] = sentiment_result[0][0]['score']
        row['neutral'] = sentiment_result[0][1]['score']
        row['positive'] = sentiment_result[0][2]['score']
        row['sarcastic'] = sarcasm_result[0][0]['score']
        row['not_sarcastic'] = sarcasm_result[0][1]['score']
    except:
        return False
    return row

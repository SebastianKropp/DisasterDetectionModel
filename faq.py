# flake8: noqa
import streamlit as st


def faq():
    st.markdown(
        """
# FAQ
## How does this model work?
When you upload a prompt, it is fed into both a Sentiment Analysis model
and a Sarcasm Detection model. The outputs of these models are then
fed within BertTDD to predict the final sentiment of the prompt.

When you ask a question, BertTDD makes use of a transformer model, 
and an attention mechanism that learns contextual relations between words (or sub-words) in a text.
In combination with pre-processed data, BertTDD can find a nuanced understandings in the text 
to discover whether the tweet is a disaster or not.


## Is my data safe?
Yes, your data is safe. BertTDD does not store your prompts or
questions. All uploaded data is deleted after you close the browser tab.

## Are the answers 100% accurate?
No
"""
    )

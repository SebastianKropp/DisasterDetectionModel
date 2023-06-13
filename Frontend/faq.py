# flake8: noqa
import streamlit as st


def faq():
    st.markdown(
        """
# FAQ
## How does this model work?

When you ask a question to BertTDD it makes use of a transformer model, 
and an attention mechanism that learns contextual relations between words (or sub-words) in a text.
In combination with pre-processed data, BertTDD can find a nuanced understandings in the text 
to discover whether the tweet is a disaster or not.

Our initial model NetTDD fed a prompt into both a Sentiment Analysis model
and a Sarcasm Detection model. The outputs of these models are then
fed within NetTDD as well additional metadata such as keyword, and location to predict the final sentiment of the prompt.\n
(* NetTDD cannot be used on live data as the Twitter API no longer viable *)

## Is my data safe?
Yes, your data is safe. BertTDD does not store your prompts or
questions. All uploaded data is deleted after you close the browser tab.

"""
    )

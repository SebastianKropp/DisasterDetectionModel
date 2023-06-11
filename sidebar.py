import streamlit as st

from faq import faq

def sidebar():
    with st.sidebar:
        st.markdown(
            "## How to use\n"
            "1. Enter a prompt (ideally from Twitter)\n"  # noqa: E501
            "2. Optionally enter location\n"
            "3. Click submit!\n"
        )
        st.markdown("---")
        st.markdown("# About")
        st.markdown(
            "BertTDD will help you to find whether the tweet is a disaster or not. Unfortunately because of recent changes with Twitter's API, we are unable to provide a full demo based on twitter links. "
        )
        st.markdown(
            "This tool is a work in progress. "
            "You can see the project on [GitHub](https://github.com/SebastianKropp/DisasterDetectionModel)"
        )
        
        faq()
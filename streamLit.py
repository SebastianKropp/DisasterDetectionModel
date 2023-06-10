import pandas as pd
import streamlit as st
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset

# Load the model
#net_model = torch.load('save_models/model.pt')

st.title('Tweet Disaster Detection')

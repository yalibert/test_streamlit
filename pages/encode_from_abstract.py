from tqdm import tqdm
import ads
import pandas as pd
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, BertConfig
import numpy as np
from sklearn.model_selection import train_test_split
from safetensors.torch import save_file
from safetensors import safe_open
from transformers import get_linear_schedule_with_warmup
import time
import random
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import json
import textwrap
import matplotlib.pyplot as plt
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import streamlit as st
import io 
import os

st.set_page_config(
        page_title="arXiv embedding - Download and embed abstract from last days",
)

st.markdown("In this page, you can download and compute the embeddings of astroph.EP abstracts.")
st.markdown("The abstracts are downloaded from arXiv, from the X last days.")

################
# download model
################

tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_cased', do_lower_case=True)

num_labels = 2

model_full = BertForSequenceClassification.from_pretrained('allenai/scibert_scivocab_cased', num_labels=num_labels,
                                                            output_attentions=False, output_hidden_states=False)

max_sequence_length = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model_full.bert
model.to(device)

################################################
#Downloading the model that is stored in dropbox
################################################

MODEL_PATH = "model_checkpoint.pth"

# Function to download and save the model file locally
def download_model():
    dropbox_url = "https://www.dropbox.com/scl/fi/i03kzt85quppl8ka64c4i/model_checkpoint.pth?rlkey=u53srsucm0jwmtd2xfprpzgv3&dl=1"

    if not os.path.exists(MODEL_PATH):  # Avoid re-downloading
        st.info("Downloading model...")
        response = requests.get(dropbox_url, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            st.success("Model downloaded successfully!")
        else:
            st.error("Failed to download model.")
            return False
    return True

# Function to load the model from the local file
@st.cache_resource
def load_model(_model,device):
    if not os.path.exists(MODEL_PATH):
        return None, None

    # Load state dict
    state_dict = torch.load(MODEL_PATH, map_location=device)

    # Handle cases where state_dict is stored under a key
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model, device

if download_model():  # Ensure the model is downloaded before loading
    model, device = load_model(model,device)

    if model:
        st.success(f"Model loaded successfully on {device}!")
    else:
        st.error("Could not load the model.")


################################
# function to compute embeddings
################################

def compute_embeddings(list_abstracts,model):
    

# Tokenize each inner list of texts
    tokenized_texts = [tokenizer.encode(text, padding='max_length', max_length=max_sequence_length,truncation=True, return_tensors="pt").squeeze() for text in list_abstracts]

# Pad each tensor in the list to ensure they are all the same length
    padded_texts = [
        torch.cat([tensor, torch.zeros(max_sequence_length - tensor.size(0), dtype=torch.long)]) for tensor in tokenized_texts
    ]


    tokenized_texts = torch.stack(tokenized_texts)
    print(tokenized_texts.shape)  # Should print (batch_size, 3, m
# attention masks    
    att_masks = []
    
    for ids in tokenized_texts:
        masks = [int(element > 0) for element in ids]
        att_masks.append(masks)
        
    padded_lists = [[mask + [0] * (max_sequence_length - len(mask)) for mask in att_masks]]
    att_masks = np.array(padded_lists).squeeze()
    att_masks = torch.Tensor(att_masks)
# computing the embeddings
    text = tokenized_texts
    mask = att_masks
    embedding = model(text, attention_mask=mask).last_hidden_state[:, 0]  

    return embedding.detach().numpy()

###########################################
# function to download abstracts from arXiv
###########################################

def fetch_recent_arxiv_abstracts(domain="astro-ph.EP", days=1, max_results=200):
    # Base URL for arXiv API
    base_url = "http://export.arxiv.org/api/query"
    
    # Query parameters
    params = {
        "search_query": domain,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    
    # Fetch data from arXiv API
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        print("Failed to fetch data.")
        return []
    
    # Parse the XML response
    root = ET.fromstring(response.content)
    
    # Namespace for arXiv XML
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    
    # Calculate date N days ago
    date_limit = datetime.now() - timedelta(days=days)
    
    # Extract and filter abstracts
    abstracts = []
    titles = []
    for entry in root.findall("atom:entry", ns):
        # Extract publication date
        published_date = entry.find("atom:published", ns).text.strip()
        published_date = datetime.fromisoformat(published_date[:-1])  # Remove 'Z' and parse
        
        # Include only entries within the date range
        if published_date >= date_limit:
            title = entry.find("atom:title", ns).text.strip()
            abstract = entry.find("atom:summary", ns).text.strip()
            #print(f"Title: {title}")
            #print(f"Published Date: {published_date}")
            #print(f"Abstract: {abstract}\n")
            abstracts.append(abstract)
            titles.append(title)
    
    return abstracts,titles

# Normalize function (min-max normalization)
def normalize(arr):
    return (arr - arr.min(axis=1, keepdims=True)) / (arr.max(axis=1, keepdims=True) - arr.min(axis=1, keepdims=True))

def format_title(text, width=20):
    return "\n".join(textwrap.wrap(text, width))

days = st.number_input("From how many days do you want to dowload abstracts?", min_value=0, max_value=100, step=1, value=3, format="%d")
st.write(f"You entered: {days}")

if days:
    recent_abstracts,recent_titles = fetch_recent_arxiv_abstracts(days=days)
    
    st.write(f"{len(recent_abstracts)} abstracts found")

    recent_abstracts_embeddings = compute_embeddings(recent_abstracts,model)

    N = len(recent_abstracts)

    normalized_array = normalize(recent_abstracts_embeddings)

    # Streamlit App Title
    st.title("Abstract Embeddings Visualization")

# Generate the plots in Streamlit
    fig_width = 4
    fig_height = 4
    for i in range(N):
        matrix = normalized_array[i].reshape(24, 32)  # Reshape

    # Create figure with fixed size
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.imshow(matrix, cmap='gray', interpolation='nearest')  # Display as B/W image
        ax.axis('off')  # Hide axes

    # Format title
        title_text = recent_titles[i]  # Directly using recent_titles[i]
        fig.suptitle(title_text, fontsize=12, fontweight='bold', ha='center')

    # Adjust layout
        fig.subplots_adjust(top=0.85, bottom=0.1)
        fig.tight_layout(pad=0)

    # Display the figure in Streamlit
        st.pyplot(fig)

    # Save the array to a .npy file in memory
    buffer = io.BytesIO()
    np.save(buffer, recent_abstracts_embeddings)
    buffer.seek(0)  # Move to the beginning of the buffer

# Create a download button
    st.download_button(
        label="Download embeddings as a NumPy Array",
        data=buffer,
        file_name="embeddings.npy",
        mime="application/octet-stream"
    )   



import streamlit as st
import torch
import requests
import os
import numpy as np
import textwrap
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification, logging
import io

# --------------------------------------------
# 0. Settings and Suppress Transformers Warnings
# --------------------------------------------
logging.set_verbosity_error()  # Suppress Hugging Face warnings

# --------------------------------------------
# 1. Download Model Weights from Dropbox if Needed
# --------------------------------------------
MODEL_PATH = "model_checkpoint.pth"
DROPBOX_URL = "https://www.dropbox.com/scl/fi/i03kzt85quppl8ka64c4i/model_checkpoint.pth?rlkey=u53srsucm0jwmtd2xfprpzgv3&dl=1"

def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model weights from Dropbox...")
        response = requests.get(DROPBOX_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            st.success("Model downloaded successfully!")
        else:
            st.error("Failed to download model weights.")
            return False
    return True

# --------------------------------------------
# 2. Load Model (only once) using session_state caching
# --------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    # Download model weights if needed
    if not download_model():
        st.stop()  # Stop execution if model download fails

    # Initialize tokenizer and model.
    tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_cased', do_lower_case=True)
    model_full = BertForSequenceClassification.from_pretrained(
        'allenai/scibert_scivocab_cased',
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )
    # We only need the underlying BERT encoder for embeddings.
    model = model_full.bert
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the checkpoint weights (using strict=False to bypass missing classifier weights)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return tokenizer, model, device

# Load and cache the model in session_state
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer, st.session_state.model, st.session_state.device = load_model()

# --------------------------------------------
# 📌 1. Load model ONCE using session state
# --------------------------------------------
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_cased', do_lower_case=True)
    model_full = BertForSequenceClassification.from_pretrained('allenai/scibert_scivocab_cased', num_labels=2)
    model = model_full.bert
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return tokenizer, model, device

if "model" not in st.session_state:
    st.session_state.tokenizer, st.session_state.model, st.session_state.device = load_model()

# --------------------------------------------
# 📌 2. Function to Download arXiv Abstracts
# --------------------------------------------
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

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

# --------------------------------------------
# 📌 3. Function to Compute Embeddings
# --------------------------------------------

def compute_embeddings(list_abstracts):
    tokenizer = st.session_state.tokenizer
    model = st.session_state.model
    device = st.session_state.device

    max_sequence_length = 512
    tokenized_texts = [tokenizer.encode(text, padding='max_length', max_length=max_sequence_length, truncation=True, return_tensors="pt").squeeze() for text in list_abstracts]
    
    tokenized_texts = torch.stack(tokenized_texts)
    att_masks = (tokenized_texts > 0).float()  # Attention masks
    tokenized_texts, att_masks = tokenized_texts.to(device), att_masks.to(device)
    
    with torch.no_grad():
        embedding = model(tokenized_texts, attention_mask=att_masks).last_hidden_state[:, 0]
    return embedding.cpu().numpy()

# --------------------------------------------
# 📌 4. Utility functions
# --------------------------------------------

def normalize(arr):
    return (arr - arr.min(axis=1, keepdims=True)) / (arr.max(axis=1, keepdims=True) - arr.min(axis=1, keepdims=True))

def format_title(text, width=20):
    return "\n".join(textwrap.wrap(text, width))


# --------------------------------------------
# 📌 5. User Input Loop
# --------------------------------------------
st.title("arXiv Abstract Embeddings")

st.write("Enter the number of days to fetch abstracts. Enter `0` to stop.")

days = st.number_input("Number of days", min_value=0, max_value=100, step=1, value=3, format="%d")

if days > 0:
    with st.spinner(f"Fetching abstracts from the last {days} days..."):
        abstracts, titles = fetch_recent_arxiv_abstracts(days=days)

    st.write(f"✅ Found **{len(abstracts)}** abstracts.")

    if abstracts:
        with st.spinner("Computing embeddings..."):
            embeddings = compute_embeddings(abstracts)
        
        # Normalize
        normalized_array = normalize(embeddings)

        # Plot
        for i, title in enumerate(titles):
            matrix = normalized_array[i].reshape(24, 32)
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(matrix, cmap='gray', interpolation='nearest')
            ax.axis('off')
            fig.suptitle(title, fontsize=10, ha='center')
            st.pyplot(fig)
            plt.close(fig)


        # Download button
        buffer = io.BytesIO()
        np.save(buffer, embeddings)
        buffer.seek(0)
        st.download_button("Download embeddings", data=buffer, file_name="embeddings.npy", mime="application/octet-stream")

elif days == 0:
    st.success("Exited. Refresh the page to start again.")
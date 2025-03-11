
import os
import subprocess

requirements_file = "requirements.txt"

if not os.path.exists("venv"):  # Check if a virtual env exists
    subprocess.run(["pip", "install", "-r", requirements_file])

import streamlit as st

st.markdown("# embedding  app")

st.markdown("This is an app to compute and visualize sciBERT embeddings of arXiv abstracts.")                                                                              

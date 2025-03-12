import streamlit as st
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path=".",
    repo_id="Rulga/LS_chat",
    repo_type="dataset",
    token=st.secrets["HF_TOKEN"]
)
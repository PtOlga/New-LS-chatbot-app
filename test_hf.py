import os
from huggingface_hub import HfApi, login
import streamlit as st

def test_token():
    try:
        # Get token from secrets
        token = st.secrets["HF_TOKEN"]
        
        # Attempt login
        login(token=token)
        
        # Create API client
        api = HfApi()
        
        # Verify token
        user_info = api.whoami(token=token)
        print(f"Successfully logged in as: {user_info}")
        
        # Check Space repository access
        space_info = api.repo_info(
            repo_id="Rulga/New-LS-chatbot-app",
            repo_type="space"
        )
        print(f"Space info: {space_info}")
        
        # Check Dataset repository access
        dataset_info = api.repo_info(
            repo_id="Rulga/LS_chat",
            repo_type="dataset"
        )
        print(f"Dataset info: {dataset_info}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_token()
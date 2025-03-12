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
        
        # Check repository access
        repo_info = api.repo_info(
            repo_id="Rulga/LS_chat",
            repo_type="dataset"
        )
        print(f"Repository info: {repo_info}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_token()
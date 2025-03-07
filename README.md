---
title: New LS Chatbot App
emoji: 👀
colorFrom: red
colorTo: red
sdk: streamlit
sdk_version: 1.42.2
app_file: app.py
pinned: false
short_description: AI chatbot for Status.law legal services using Streamlit
---

# LS Chatbot app

It is a chat app built using Streamlit that allows users to interact with an AI model to communicate about www.Status.law

## Project Structure
```
.
├── app.py
├── chat_history/
│   └── .gitkeep
├── vector_store/
│   └── .gitkeep
└── requirements.txt
```

## Required Directories
- `chat_history/` - Stores chat history JSON files
- `vector_store/` - Stores FAISS vector database

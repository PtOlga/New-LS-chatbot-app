import os
import json
import smtplib
import streamlit as st
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from requests.exceptions import RequestException, Timeout

# Streamlit page configuration
st.set_page_config(page_title="Legal Chatbot", page_icon="🤖")

# Load environment variables
if os.path.exists(".env"):
    load_dotenv(verbose=True)

# Load API keys
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    USER_AGENT = st.secrets["USER_AGENT"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    EMAIL_SENDER = st.secrets["EMAIL_SENDER"]
    EMAIL_PASSWORD = st.secrets["EMAIL_PASSWORD"]
except FileNotFoundError:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    USER_AGENT = os.getenv("USER_AGENT")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMAIL_SENDER = os.getenv("EMAIL_SENDER")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# Check if API keys are set
if not all([GROQ_API_KEY, USER_AGENT, OPENAI_API_KEY, EMAIL_SENDER, EMAIL_PASSWORD]):
    st.error("Error: Missing required environment variables.")
    st.stop()

# Initialize LLM
try:
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.6, api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"LLM initialization failed: {e}")
    st.stop()

# Initialize embeddings model
embeddings_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")

# List of website pages for knowledge base
urls = [
    "https://status.law",
    "https://status.law/about",
    "https://status.law/careers",
    "https://status.law/challenging-sanctions",
    "https://status.law/contact",
    "https://status.law/cross-border-banking-legal-issues",
    "https://status.law/extradition-defense",
    "https://status.law/international-prosecution-protection",
    "https://status.law/interpol-red-notice-removal",
    "https://status.law/practice-areas",
    "https://status.law/reputation-protection",
    "https://status.law/faq"
]

# Path to knowledge base storage
VECTOR_STORE_PATH = "storage/vector_store"
HISTORY_PATH = "storage/chat_history.json"

# Function to build knowledge base
def build_knowledge_base():
    documents = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            documents.extend(loader.load())
            st.write(f"[INFO] Loaded content from {url}")
        except (RequestException, Timeout) as e:
            st.write(f"[ERROR] Failed to load {url}: {e}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    st.write(f"[INFO] Split into {len(chunks)} chunks")

    vector_store = FAISS.from_documents(chunks, embeddings_model)
    vector_store.save_local(VECTOR_STORE_PATH)

    st.write("[INFO] Knowledge base successfully created and saved")
    return vector_store

# Function to load existing knowledge base
def load_knowledge_base():
    if os.path.exists(VECTOR_STORE_PATH):
        return FAISS.load_local(
            VECTOR_STORE_PATH, 
            embeddings_model, 
            allow_dangerous_deserialization=True 
        )
    return None

# Function to load chat history
def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return []

# Function to save chat history
def save_history(history):
    with open(HISTORY_PATH, "w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=4)

# Function to send chat history via email
def send_email(recipient_email, subject, message):
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["To"] = recipient_email
        msg["Subject"] = subject
        msg.attach(MIMEText(message, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        return True
    except Exception as e:
        st.error(f"Email sending error: {e}")
        return False

# Load or create knowledge base
if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_knowledge_base()

# Prompt to create knowledge base if missing
if st.session_state.vector_store is None:
    st.write("Knowledge base not found. Click the button to generate it.")
    if st.button("Generate Knowledge Base"):
        with st.spinner("Building knowledge base..."):
            st.session_state.vector_store = build_knowledge_base()
            st.success("Knowledge base successfully created!")
            st.rerun()
else:
    st.write("Knowledge base loaded. You can ask questions.")

# Chatbot prompt template
template = """
You are a helpful legal assistant answering questions based on information from status.law.
Answer accurately and concisely.
Question: {question}
Only use the provided context to answer the question.
Context: {context}
"""
prompt = PromptTemplate.from_template(template)

# Initialize processing chain
if "chain" not in st.session_state:
    st.session_state.chain = (
        RunnableLambda(lambda x: {"context": x["context"], "question": x["question"]})
        | prompt
        | llm
        | StrOutputParser()
    )

chain = st.session_state.chain

# Load chat history
if "message_history" not in st.session_state:
    st.session_state.message_history = load_history()

# Chat input
user_input = st.text_input("Enter your question:")
if st.button("Send") and user_input:
    vector_store = st.session_state.vector_store
    retrieved_docs = vector_store.similarity_search(user_input)
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    response = chain.invoke({"question": user_input, "context": context_text})

    # Save to session and persist history
    st.session_state.message_history.append({"question": user_input, "answer": response})
    save_history(st.session_state.message_history)

    st.write(response)

# Display chat history
if st.session_state.message_history:
    st.write("### Chat History")
    for msg in st.session_state.message_history:
        st.write(f"**User:** {msg['question']}")
        st.write(f"**Bot:** {msg['answer']}")

# Email history feature
recipient_email = st.text_input("Enter email to receive chat history:")
if st.button("Send History via Email"):
    if st.session_state.message_history:
        history_text = "\n\n".join([f"User: {msg['question']}\nBot: {msg['answer']}" for msg in st.session_state.message_history])
        success = send_email(recipient_email, "Chat History", history_text)
        if success:
            st.success(f"Chat history sent to {recipient_email}!")
    else:
        st.warning("Chat history is empty.")

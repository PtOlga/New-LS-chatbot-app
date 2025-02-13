import os
import streamlit as st
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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Page configuration
st.set_page_config(page_title="Legal Chatbot", page_icon="🤖")

# Load environment variables
if os.path.exists(".env"):
    load_dotenv(verbose=True)

# Load API keys and credentials from secrets (Hugging Face)
EMAIL_SENDER = st.secrets["EMAIL_SENDER"]
EMAIL_PASSWORD = st.secrets["EMAIL_PASSWORD"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
USER_AGENT = st.secrets["USER_AGENT"]
LANGSMITH_TRACING = st.secrets["LANGSMITH_TRACING"]
LANGSMITH_ENDPOINT = st.secrets["LANGSMITH_ENDPOINT"]
LANGSMITH_API_KEY = st.secrets["LANGSMITH_API_KEY"]
LANGSMITH_PROJECT = st.secrets["LANGSMITH_PROJECT"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Check API keys
if not all([GROQ_API_KEY, USER_AGENT, LANGSMITH_TRACING, LANGSMITH_ENDPOINT, LANGSMITH_API_KEY, LANGSMITH_PROJECT, OPENAI_API_KEY]):
    st.error("Error: Not all environment variables are set.")
    st.stop()

# Initialize LLM
try:
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.6, api_key=GROQ_API_KEY)
    print("[DEBUG] LLM successfully initialized")
except Exception as e:
    st.error(f"LLM initialization error: {e}")
    st.stop()

# Initialize embeddings model
embeddings_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
print("[DEBUG] Embeddings model loaded")

# List of URLs for knowledge base
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

# Path to vector store
VECTOR_STORE_PATH = "vector_store"

# Function to build knowledge base
def build_knowledge_base():
    documents = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            documents.extend(loader.load())
            st.write(f"[DEBUG] Loaded content from {url}")
        except (RequestException, Timeout) as e:
            st.write(f"[ERROR] Error loading page {url}: {e}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    st.write(f"[DEBUG] Split into {len(chunks)} chunks")
    vector_store = FAISS.from_documents(chunks, embeddings_model)
    vector_store.save_local(VECTOR_STORE_PATH)
    st.write("[DEBUG] Vector store created and saved")
    return vector_store

# Function to load knowledge base
def load_knowledge_base():
    if os.path.exists(VECTOR_STORE_PATH):
        st.write("[DEBUG] Loading existing vector store")
        return FAISS.load_local(
            VECTOR_STORE_PATH, 
            embeddings_model, 
            allow_dangerous_deserialization=True 
        )
    else:
        st.write("[DEBUG] Vector store not found")
        return None

# Load the knowledge base
if "vector_store" not in st.session_state or st.session_state.vector_store is None:
    vector_store = load_knowledge_base()
    if vector_store is None:
        st.write("Knowledge base not found. Press the button to create it.")
        if st.button("Create Knowledge Base"):
            with st.spinner("Creating knowledge base..."):
                vector_store = build_knowledge_base()
                st.session_state.vector_store = vector_store
                st.success("Knowledge base successfully created!")
                st.rerun()  # Restart app to switch to Q&A mode
    else:
        st.session_state.vector_store = vector_store
else:
    st.write("Knowledge base loaded. You can now ask questions.")

# If knowledge base exists, proceed with Q&A mode
if st.session_state.vector_store:
    # Prompt template for the chatbot
    template = """
    You are a helpful legal assistant that answers questions based on information from status.law.
    Answer accurately and concisely.
    Question: {question}
    Only use the provided context to answer the question.
    Context: {context}
    """
    prompt = PromptTemplate.from_template(template)

    # Initialize the chain for question-answering
    if "chain" not in st.session_state:
        st.session_state.chain = (
            RunnableLambda(lambda x: {"context": x["context"], "question": x["question"]}) 
            | prompt
            | llm
            | StrOutputParser()
        )
    chain = st.session_state.chain

    # User input
    user_input = st.text_input("Enter your question:")
    if st.button("Send") and user_input:
        # Ensure vector_store is not None before calling similarity_search
        if st.session_state.vector_store:
            # Retrieve relevant documents from the vector store
            retrieved_docs = st.session_state.vector_store.similarity_search(user_input)
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

            # Generate the response
            response = chain.invoke({"question": user_input, "context": context_text})

            # Save to session state and message history
            if "message_history" not in st.session_state:
                st.session_state.message_history = []
            st.session_state.message_history.append({"question": user_input, "answer": response})
            save_history(st.session_state.message_history)

            st.write(response)

        else:
            st.error("Error: vector store is not loaded properly.")

    # Display message history
    if "message_history" in st.session_state:
        st.write("### Message History")
        for msg in st.session_state.message_history:
            st.write(f"**User:** {msg['question']}")
            st.write(f"**Bot:** {msg['answer']}")

# Function to send email with chat history
def send_email(subject, body, to_email):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = to_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))
        
        # SMTP connection
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            text = msg.as_string()
            server.sendmail(EMAIL_SENDER, to_email, text)

        st.success("Chat history has been sent to your email.")
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")

# Button to send email
if st.button("Send Chat History to Email"):
    if "message_history" in st.session_state and st.session_state.message_history:
        email_body = "\n\n".join([f"User: {msg['question']}\nBot: {msg['answer']}" for msg in st.session_state.message_history])
        send_email("Chat History", email_body, EMAIL_SENDER)  # Send to the same email as sender
    else:
        st.error("No chat history available to send.")

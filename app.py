import os
import time
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
import requests
import json
from datetime import datetime

# Define base directory and absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store")
CHAT_HISTORY_DIR = os.path.join(BASE_DIR, "chat_history")

# Create required directories with absolute paths
REQUIRED_DIRS = [CHAT_HISTORY_DIR, VECTOR_STORE_PATH]
for dir_path in REQUIRED_DIRS:
    os.makedirs(dir_path, exist_ok=True)
    gitkeep_path = os.path.join(dir_path, '.gitkeep')
    if not os.path.exists(gitkeep_path):
        with open(gitkeep_path, 'w') as f:
            pass

# Page configuration
st.set_page_config(page_title="Status Law Assistant", page_icon="⚖️")

# Knowledge base info in session_state
if 'kb_info' not in st.session_state:
    st.session_state.kb_info = {
        'build_time': None,
        'size': None
    }

# Initialize chat_history in session_state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize messages if not exists
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Create history folder if not exists
#if not os.path.exists("chat_history"):
#    os.makedirs("chat_history")

# Display title and knowledge base info
# st.title("www.Status.Law Legal Assistant")

st.markdown(
    '''
    <h1>
        ⚖️ 
        <a href="https://status.law/" style="text-decoration: underline; color: blue; font-size: inherit;">
            Status.Law
        </a> 
        Legal Assistant
    </h1>
    ''',
    unsafe_allow_html=True
)

if st.session_state.kb_info['build_time'] and st.session_state.kb_info['size']:
    st.caption(f"(Knowledge base build time: {st.session_state.kb_info['build_time']:.2f} seconds, "
               f"size: {st.session_state.kb_info['size']:.2f} MB)")

# Path to store vector database
# VECTOR_STORE_PATH = "vector_store"

# Website URLs
urls = [
    "https://status.law",  
    "https://status.law/about",
    "https://status.law/careers",
    "https://status.law/challenging-sanctions",
    "https://status.law/tariffs-for-services-against-extradition-en",
    "https://status.law/law-firm-contact-legal-protection",
    "https://status.law/cross-border-banking-legal-issues", 
    "https://status.law/extradition-defense", 
    "https://status.law/international-prosecution-protection", 
    "https://status.law/interpol-red-notice-removal",  
    "https://status.law/practice-areas",  
    "https://status.law/reputation-protection",
    "https://status.law/faq"
]

# Load secrets
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception as e:
    st.error("Error loading secrets. Please check your configuration.")
    st.stop()

# Initialize models
@st.cache_resource
def init_models():
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.6,
        api_key=GROQ_API_KEY
    )
    embeddings = HuggingFaceEmbeddings(
        #model_name="intfloat/multilingual-e5-large-instruct"
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return llm, embeddings

# Build knowledge base
def build_knowledge_base(embeddings):
    start_time = time.time()
    
    documents = []
    with st.status("Loading website content...") as status:
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                documents.extend(docs)
                status.update(label=f"Loaded {url}")
            except Exception as e:
                st.error(f"Error loading {url}: {str(e)}")
                
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Immediately save vector store after creation
    force_save_vector_store(vector_store)
    
    end_time = time.time()
    build_time = end_time - start_time
    
    # Calculate knowledge base size
    total_size = 0
    for path, dirs, files in os.walk(VECTOR_STORE_PATH):
        for f in files:
            fp = os.path.join(path, f)
            total_size += os.path.getsize(fp)
    size_mb = total_size / (1024 * 1024)
    
    # Save knowledge base info
    st.session_state.kb_info['build_time'] = build_time
    st.session_state.kb_info['size'] = size_mb
    
    st.success(f"""
    Knowledge base created successfully:
    - Time taken: {build_time:.2f} seconds
    - Size: {size_mb:.2f} MB
    - Number of chunks: {len(chunks)}
    """)
    
    return vector_store

# Function to save chat history
def save_chat_to_file(chat_history):
    """Save chat history to file using absolute path"""
    current_date = datetime.now().strftime("%Y-%m-%d")
    filename = os.path.join(CHAT_HISTORY_DIR, f"chat_history_{current_date}.json")
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Error saving chat history: {e}")

# Function to load chat history
def load_chat_history():
    """Load chat history from file using absolute path"""
    current_date = datetime.now().strftime("%Y-%m-%d")
    filename = os.path.join(CHAT_HISTORY_DIR, f"chat_history_{current_date}.json")
    
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading chat history: {e}")
            return []
    return []

def check_directory_permissions(directory):
    """Check if directory has proper read/write permissions"""
    try:
        # Check if directory exists and create if not
        os.makedirs(directory, exist_ok=True)
        
        # Try to create a test file
        test_file = os.path.join(directory, "write_test.txt")
        with open(test_file, "w") as f:
            f.write("test")
            f.flush()
            os.fsync(f.fileno())  # Force write to disk
            
        # Try to read the test file
        with open(test_file, "r") as f:
            content = f.read()
            if content != "test":
                raise Exception("File content verification failed")
                
        # Clean up
        os.remove(test_file)
        
        return True, None
        
    except Exception as e:
        permissions = oct(os.stat(directory).st_mode)[-3:] if os.path.exists(directory) else "N/A"
        error_msg = f"Permission error: {str(e)} (Directory permissions: {permissions})"
        return False, error_msg

def force_save_vector_store(vector_store):
    """Ensures vector store is properly saved to disk"""
    try:
        # Check directory permissions
        success, error_msg = check_directory_permissions(VECTOR_STORE_PATH)
        if not success:
            raise Exception(error_msg)
        
        # Save vector store
        vector_store.save_local(VECTOR_STORE_PATH)
        
        # Verify vector store files were created
        index_file = os.path.join(VECTOR_STORE_PATH, "index.faiss")
        if not os.path.exists(index_file):
            raise Exception("Vector store files were not created")
            
        # Verify file permissions
        if not os.access(index_file, os.R_OK | os.W_OK):
            raise Exception(f"Insufficient permissions for vector store files")
            
        #st.caption("✅ Vector store saved successfully")
        st.toast("✅ Vector store saved", icon="💾")
        
    except Exception as e:
        error_msg = f"❌ Failed to save vector store: {str(e)}"
        st.caption(error_msg)
        st.error(error_msg)  # Also show as error message
        raise Exception(error_msg)

def force_save_chat_history(chat_entry):
    """Ensures chat history is properly saved to disk"""
    try:
        # Check directory permissions
        success, error_msg = check_directory_permissions(CHAT_HISTORY_DIR)
        if not success:
            raise Exception(error_msg)
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        filename = os.path.join(CHAT_HISTORY_DIR, f"chat_history_{current_date}.json")
        
        # Load existing history
        existing_history = []
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                existing_history = json.load(f)
        
        # Add new entry
        existing_history.append(chat_entry)
        
        # Save updated history with fsync to ensure disk write
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(existing_history, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
            os.sync()  # Принудительная синхронизация файловой системы
            
        # Verify file was created and is readable
        if not os.path.exists(filename):
            raise Exception("Chat history file was not created")
            
        if not os.access(filename, os.R_OK | os.W_OK):
            raise Exception(f"Insufficient permissions for chat history file")
            
        #st.caption("✅ Chat history saved successfully")
        st.toast("✅ Chat history saved", icon="💬")
        
    except Exception as e:
        error_msg = f"❌ Failed to save chat history: {str(e)}"
        st.caption(error_msg)
        st.error(error_msg)  # Also show as error message
        raise Exception(error_msg)

# Main function
def main():
    # Initialize models
    llm, embeddings = init_models()
    
    # Check if knowledge base exists
    if not os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
        st.warning("Knowledge base not found. Please create it first.")
        if st.button("Create Knowledge Base"):
            with st.spinner("Creating knowledge base... This may take a few minutes."):
                try:
                    vector_store = build_knowledge_base(embeddings)
                    st.session_state.vector_store = vector_store
                    st.success("Knowledge base created successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error creating knowledge base: {e}")
        return
    
    # Load existing knowledge base
    if 'vector_store' not in st.session_state:
        try:
            st.session_state.vector_store = FAISS.load_local(
                VECTOR_STORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.error(f"Error loading knowledge base: {e}")
            return
        
    with st.sidebar:
        st.write(f"Working directory: {BASE_DIR}")
        st.write(f"Vector store: {VECTOR_STORE_PATH}")
        st.write(f"Chat history: {CHAT_HISTORY_DIR}")
    
    # Chat mode
    if 'vector_store' in st.session_state:
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Load chat history on startup
        if not st.session_state.chat_history:
            st.session_state.chat_history = load_chat_history()
        
        # Display chat history
        for message in st.session_state.messages:
            st.chat_message("user").write(message["question"])
            st.chat_message("assistant").write(message["answer"])
        
        # User input
        if question := st.chat_input("Ask your question"):
            st.chat_message("user").write(question)
            
            # Retrieve context and generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    context = st.session_state.vector_store.similarity_search(question)
                    context_text = "\n".join([doc.page_content for doc in context])
                    
                    prompt = PromptTemplate.from_template("""
You are a helpful and polite legal assistant at Status Law, an international law firm specializing in extradition cases.

Answer in the language in which the question was asked.

Use the following information to answer questions:
- Primary context: {context}
- Services and pricing page: https://status.law/tariffs-for-services-against-extradition-en

When asked about service prices or specific legal services:
1. Search for the specific service on our website
2. Provide a brief description of how Status Law can help with this specific issue
3. Explain the key benefits or features of this service
4. Only share the direct link to pricing (https://status.law/tariffs-for-services-against-extradition-en) if the question is specifically about prices
5. For general service inquiries without price questions, focus on service descriptions without sharing the pricing page link

For example:
- If asked "How much does legal representation in court cost?", describe the service briefly and provide the pricing page link
- If asked "Can you help with document preparation?", explain the service without sharing the pricing link

If you cannot answer based on the available information, say so politely and offer to contact Status Law directly via the following channels:
- For all users: +32465594521 (landline phone)
- For English and Swedish speakers only: +46728495129 (available on WhatsApp, Telegram, Signal, IMO)
- Provide a link to the contact form: [Contact Form](https://status.law/law-firm-contact-legal-protection/)

Question: {question}

Response Guidelines:
1. Answer in the user's language
2. Be concise but informative
3. Cite specific service details when relevant
4. Emphasize our international expertise in extradition law
5. Share pricing page link ONLY when questions are specifically about costs
6. Offer contact options if the question requires detailed legal advice
""")
                    
                    chain = prompt | llm | StrOutputParser()
                    response = chain.invoke({
                        "context": context_text,
                        "question": question
                    })
                    
                    st.write(response)
                    
                    # Create chat entry
                    chat_entry = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "question": question,
                        "answer": response,
                        "context": context_text
                    }
                    
                    # Force save chat history
                    force_save_chat_history(chat_entry)
                    
                    # Update session state
                    if "chat_history" not in st.session_state:
                        st.session_state.chat_history = []
                    st.session_state.chat_history.append(chat_entry)
                    
                    st.session_state.messages.append({
                        "question": question,
                        "answer": response
                    })

if __name__ == "__main__":
     main()

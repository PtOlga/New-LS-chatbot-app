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

# Page configuration
st.set_page_config(page_title="Status Law Assistant", page_icon="⚖️")

# Knowledge base info in session_state
if 'kb_info' not in st.session_state:
    st.session_state.kb_info = {
        'build_time': None,
        'size': None
    }

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
VECTOR_STORE_PATH = "vector_store"

# Создание папки истории, если она не существует
if not os.path.exists("chat_history"):
    os.makedirs("chat_history")

# Website URLs
urls = [
    "https://status.law",  
    "https://status.law/about",
    "https://status.law/careers",
    "https://status.law/challenging-sanctions",
    "https://status.law/tariffs-for-services-against-extradition-en",
    "https://status.law/law-firm-contact-legal-protection"
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
        model_name="intfloat/multilingual-e5-large-instruct"
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
    vector_store.save_local(VECTOR_STORE_PATH)
    
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

# Функция для сохранения истории чата
def save_chat_to_file(chat_history):
    current_date = datetime.now().strftime("%Y-%m-%d")
    filename = f"chat_history/chat_history_{current_date}.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Ошибка при сохранении истории чата: {e}")

# Функция для загрузки истории чата
def load_chat_history():
    current_date = datetime.now().strftime("%Y-%m-%d")
    filename = f"chat_history/chat_history_{current_date}.json"
    
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Ошибка при загрузке истории чата: {e}")
            return []
    return []

# Main function
def main():
    # Initialize models
    llm, embeddings = init_models()
    
    # Check if knowledge base exists
    if not os.path.exists(VECTOR_STORE_PATH):
        st.warning("Knowledge base not found.")
        if st.button("Create Knowledge Base"):
            vector_store = build_knowledge_base(embeddings)
            st.session_state.vector_store = vector_store
            st.rerun()
    else:
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = FAISS.load_local(
                VECTOR_STORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
    
    # Chat mode
    if 'vector_store' in st.session_state:
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Загрузка истории чата при запуске
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
                    You are a helpful and polite legal assistant at Status Law.
                    You answer in the language in which the question was asked.
                    Answer the question based on the context provided.
                    If you cannot answer based on the context, say so politely and offer to contact Status Law directly via the following channels:
                    - For all users: +32465594521 (landline phone).
                    - For English and Swedish speakers only: +46728495129 (available on WhatsApp, Telegram, Signal, IMO).
                    - Provide a link to the contact form: [Contact Form](https://status.law/law-firm-contact-legal-protection/).
                    Answer professionally but in a friendly manner.

                    Example:
                    Q: How can I challenge the sanctions?
                    A: To challenge the sanctions, you should consult with our legal team, who specialize in this area. Please contact us directly for detailed advice. You can fill out our contact form here: [Contact Form](https://status.law/law-firm-contact-legal-protection/).

                    Context: {context}
                    Question: {question}
                    """)
                    
                    chain = prompt | llm | StrOutputParser()
                    response = chain.invoke({
                        "context": context_text,
                        "question": question
                    })
                    
                    st.write(response)
                    
                    # Сохранение в историю чата
                    chat_entry = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "question": question,
                        "answer": response,
                        "context": context_text
                    }
                    
                    st.session_state.chat_history.append(chat_entry)
                    save_chat_to_file(st.session_state.chat_history)
                    
                    st.session_state.messages.append({
                        "question": question,
                        "answer": response
                    })

if __name__ == "__main__":
    main()

# Добавить кнопку для выгрузки истории чата (опционально)
if st.sidebar.button("Скачать историю чата"):
    current_date = datetime.now().strftime("%Y-%m-%d")
    filename = f"chat_history_{current_date}.json"
    
    if st.session_state.chat_history:
        json_str = json.dumps(st.session_state.chat_history, ensure_ascii=False, indent=2)
        st.download_button(
            label="Скачать JSON",
            data=json_str.encode('utf-8'),
            file_name=filename,
            mime="application/json"
    main()

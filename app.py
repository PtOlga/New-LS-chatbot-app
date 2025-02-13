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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Базовая конфигурация страницы
st.set_page_config(page_title="Legal Assistant", page_icon="⚖️")
st.title("Legal Assistant")

# Путь для хранения базы знаний
VECTOR_STORE_PATH = "vector_store"

# URLs вашего сайта
urls = [
    "https://status.law",  
    "https://status.law/about",
    # ... остальные URLs ...
]

# Загрузка секретов
try:
    EMAIL_SENDER = st.secrets["EMAIL_SENDER"]
    EMAIL_PASSWORD = st.secrets["EMAIL_PASSWORD"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception as e:
    st.error("Error loading secrets. Please check your configuration.")
    st.stop()

# Инициализация моделей
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

# Создание базы знаний
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
    
    st.success(f"""
    Knowledge base created successfully:
    - Time taken: {build_time:.2f} seconds
    - Number of chunks: {len(chunks)}
    """)
    
    return vector_store

# Отправка email
def send_chat_history(history):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_SENDER
        msg['Subject'] = "Chat History Update"
        
        body = "\n\n".join([
            f"Q: {item['question']}\nA: {item['answer']}"
            for item in history
        ])
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")

# Основной код
def main():
    # Инициализация моделей
    llm, embeddings = init_models()
    
    # Проверка существования базы знаний
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
    
    # Режим чата
    if 'vector_store' in st.session_state:
        if 'messages' not in st.session_state:
            st.session_state.messages = []
            
        # Показ истории сообщений
        for message in st.session_state.messages:
            st.chat_message("user").write(message["question"])
            st.chat_message("assistant").write(message["answer"])
            
        # Ввод пользователя
        if question := st.chat_input("Ask your question"):
            st.chat_message("user").write(question)
            
            # Поиск контекста и генерация ответа
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    context = st.session_state.vector_store.similarity_search(question)
                    context_text = "\n".join([doc.page_content for doc in context])
                    
                    prompt = PromptTemplate.from_template("""
                    You are a helpful and polite legal assistant. Answer the question based on the provided context.
                    If you cannot answer based on the context, say so politely.
                    
                    Context: {context}
                    Question: {question}
                    """)
                    
                    chain = prompt | llm | StrOutputParser()
                    response = chain.invoke({
                        "context": context_text,
                        "question": question
                    })
                    
                    st.write(response)
                    
                    # Сохранение истории
                    st.session_state.messages.append({
                        "question": question,
                        "answer": response
                    })
                    
                    # Отправка email
                    send_chat_history(st.session_state.messages)

if __name__ == "__main__":
    main()
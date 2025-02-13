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

# Установка конфигурации страницы
st.set_page_config(page_title="Legal Chatbot", page_icon="🤖")

# Загрузка переменных окружения
if os.path.exists(".env"):
    load_dotenv(verbose=True)

# Загрузка API-ключей
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    USER_AGENT = st.secrets["USER_AGENT"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except FileNotFoundError:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    USER_AGENT = os.getenv("USER_AGENT")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Проверка API-ключей
if not all([GROQ_API_KEY, USER_AGENT, OPENAI_API_KEY]):
    st.error("Ошибка: Не все переменные окружения заданы.")
    st.stop()

# Инициализация LLM
try:
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.6, api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"Ошибка инициализации LLM: {e}")
    st.stop()

# Инициализация эмбеддингов
embeddings_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")

# Список страниц для анализа
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

# Путь к файлу векторного хранилища
VECTOR_STORE_PATH = "vector_store"

# Функция для создания базы знаний
def build_knowledge_base():
    documents = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            documents.extend(loader.load())
        except (RequestException, Timeout) as e:
            st.write(f"[ERROR] Ошибка загрузки страницы {url}: {e}")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    vector_store = FAISS.from_documents(chunks, embeddings_model)
    vector_store.save_local(VECTOR_STORE_PATH)
    
    return vector_store

# Функция для загрузки базы знаний
def load_knowledge_base():
    if os.path.exists(VECTOR_STORE_PATH):
        return FAISS.load_local(
            VECTOR_STORE_PATH, 
            embeddings_model, 
            allow_dangerous_deserialization=True 
        )
    return None

# Загружаем базу, если её нет в `st.session_state`
if "vector_store" not in st.session_state or st.session_state.vector_store is None:
    st.session_state.vector_store = load_knowledge_base()

vector_store = st.session_state.vector_store

# Если база знаний отсутствует, предлагаем её создать
if vector_store is None:
    st.write("База знаний не найдена. Нажмите кнопку, чтобы создать её.")
    if st.button("Создать базу знаний"):
        with st.spinner("Создание базы знаний..."):
            st.session_state.vector_store = build_knowledge_base()
            st.success("База знаний успешно создана!")
            st.rerun()  # Перезапуск приложения
else:
    st.write("База знаний загружена. Вы можете задать вопрос.")

    # Промпт для бота
    template = """
    You are a helpful legal assistant that answers questions based on information from status.law.
    Answer accurately and concisely.
    Question: {question}
    Only use the provided context to answer the question.
    Context: {context}
    """
    prompt = PromptTemplate.from_template(template)

    # Инициализация цепочки обработки запроса
    if "chain" not in st.session_state:
        st.session_state.chain = (
            RunnableLambda(lambda x: {"context": x["context"], "question": x["question"]}) 
            | prompt
            | llm
            | StrOutputParser()
        )
    chain = st.session_state.chain

    # Поле для ввода вопроса
    user_input = st.text_input("Введите ваш вопрос:")
    if st.button("Отправить") and user_input:
        if st.session_state.vector_store:  # Проверяем, что база знаний загружена
            retrieved_docs = st.session_state.vector_store.similarity_search(user_input)
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

            # Генерация ответа
            response = chain.invoke({"question": user_input, "context": context_text})

            # Сохранение истории сообщений
            if "message_history" not in st.session_state:
                st.session_state.message_history = []
            st.session_state.message_history.append({"question": user_input, "answer": response})

            # Вывод ответа
            st.write(response)
        else:
            st.error("Ошибка: база знаний не загружена.")

    # Вывод истории сообщений
    if "message_history" in st.session_state:
        st.write("### История сообщений")
        for msg in st.session_state.message_history:
            st.write(f"**User:** {msg['question']}")
            st.write(f"**Bot:** {msg['answer']}")

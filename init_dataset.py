import os
from huggingface_hub import HfApi, create_repo
import json

# Конфигурация
HF_TOKEN = "your_token_here"  # Замените на ваш токен
HF_USERNAME = "Rulga"
DATASET_NAME = "LS_chat"
DATASET_REPO = f"{HF_USERNAME}/{DATASET_NAME}"

# Создаем временную структуру
temp_dir = "temp_dataset"
os.makedirs(os.path.join(temp_dir, "chat_history"), exist_ok=True)
os.makedirs(os.path.join(temp_dir, "vector_store"), exist_ok=True)

# Создаем пустые .gitkeep файлы
with open(os.path.join(temp_dir, "chat_history", ".gitkeep"), "w") as f:
    pass
with open(os.path.join(temp_dir, "vector_store", ".gitkeep"), "w") as f:
    pass

# Создаем README.md с описанием структуры
readme_content = """
# LS Chat Dataset

This dataset contains chat history and vector store for the Status.Law Legal Assistant.

## Structure

- `chat_history/`: Contains daily chat history files
- `vector_store/`: Contains FAISS vector store files

## Usage

This dataset is automatically updated by the Status.Law Legal Assistant application.
"""

with open(os.path.join(temp_dir, "README.md"), "w") as f:
    f.write(readme_content)

# Инициализируем и загружаем на Hugging Face
try:
    api = HfApi()
    
    # Создаем репозиторий, если он не существует
    try:
        api.repo_info(repo_id=DATASET_REPO, repo_type="dataset")
        print(f"Repository {DATASET_REPO} already exists")
    except Exception:
        create_repo(DATASET_REPO, repo_type="dataset", token=HF_TOKEN)
        print(f"Created new repository {DATASET_REPO}")
    
    # Загружаем структуру
    api.upload_folder(
        folder_path=temp_dir,
        repo_id=DATASET_REPO,
        repo_type="dataset",
        commit_message="Initialize dataset structure",
        token=HF_TOKEN
    )
    print("Successfully initialized dataset structure!")

except Exception as e:
    print(f"Error: {str(e)}")

finally:
    # Очищаем временные файлы
    import shutil
    shutil.rmtree(temp_dir)
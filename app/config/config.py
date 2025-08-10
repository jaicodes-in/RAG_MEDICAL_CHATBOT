import os

HF_TOKEN=os.getenv("HF_TOKEN")

HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct"
DB_FAISS_PATH='vector/db_faiss'
DATA_PATH='data/'

CHUNK_SIZE=500
CHUNK_OVERLAP=50

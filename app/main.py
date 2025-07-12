from fastapi import FastAPI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# rag 부분 코드(나중에 파일 분리 예정)
from langchain.document_loaders import TextLoader
import os

# 현재 파일(app/main.py) 기준 상위 디렉토리 → data/dummy.txt
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "data", "dummy.txt")

loader = TextLoader(file_path, encoding='utf-8')
docs = loader.load()

#임베딩 모델 로드
embeddings_model = HuggingFaceEmbeddings(
    model_name='jhgan/ko-sbert-nli',
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True},
)

# 문서 임베딩
embeddings = embeddings_model.encode(docs[0].page_content)
# FAISS 인덱스 생성
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 벡터 저장소 생성 ( 랭체인에 있는 벡터스토어 인터페이스를 따른다. )
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=model,
    persist_directory="data/chroma_db"
)

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

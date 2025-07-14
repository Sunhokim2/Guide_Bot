# D:\GGGGGGG\Guide_Bot\ingest_data.py

import os
from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader # TextLoader 추가
from langchain.text_splitter import RecursiveCharacterTextSplitter # 텍스트 분할기 추가

# .env 파일 로드
load_dotenv()

# 환경 변수에서 AWS 리전 로드
AWS_REGION_NAME = os.getenv("AWS_REGION_NAME") # Bedrock 모델이 지원되는 리전으로 변경

# ChromaDB 저장 경로
PERSIST_DIRECTORY = "chroma_db"
# 데이터가 있는 폴더 경로
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIRECTORY = os.path.join(BASE_DIR, "data")

# 임베딩 모델 초기화
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name=AWS_REGION_NAME
)

def ingest_documents():
    print(f"Loading documents from {DATA_DIRECTORY}...")
    documents = []
    for filename in os.listdir(DATA_DIRECTORY):
        if filename.endswith(".txt"): # 텍스트 파일만 로드
            file_path = os.path.join(DATA_DIRECTORY, filename)
            loader = TextLoader(file_path, encoding="utf-8") # 인코딩 명시
            documents.extend(loader.load())
    print(f"Loaded {len(documents)} documents.")

    if not documents:
        print(f"No text documents found in '{DATA_DIRECTORY}'. Please add .txt files to this directory.")
        return

    # 문서 분할 (Chunking)
    # RAG 성능에 매우 중요! 텍스트 길이와 오버랩을 조절하여 문맥 유지
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 한 청크의 최대 문자 수
        chunk_overlap=200 # 청크 간 겹치는 문자 수 (문맥 유지)
    )
    splits = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(splits)} chunks.")

    # ChromaDB 생성 및 임베딩 저장
    print(f"Creating ChromaDB at {PERSIST_DIRECTORY} and embedding documents...")
    try:
        # 기존 DB가 있다면 삭제 후 새로 생성하거나, update 방식 고려
        # 단순 재구축 시에는 매번 삭제 후 새로 만드는 것이 확실합니다.
        if os.path.exists(PERSIST_DIRECTORY):
            import shutil
            shutil.rmtree(PERSIST_DIRECTORY)
            print(f"Existing ChromaDB at {PERSIST_DIRECTORY} removed.")

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )

        print("ChromaDB created and documents embedded successfully.")
    except Exception as e:
        print(f"Error creating ChromaDB: {e}")

if __name__ == "__main__":
    ingest_documents()
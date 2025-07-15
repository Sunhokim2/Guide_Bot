import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# PGVector 클래스 임포트
from langchain_community.vectorstores import PGVector
import psycopg2 # DB 연결 테스트용

PG_CONNECTION_STRING = os.getenv("PG_CONNECTION_STRING")

# 데이터가 있는 폴더 경로
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIRECTORY = os.path.join(BASE_DIR, "data")

# Pgvector 설정
COLLECTION_NAME = "guide_bot_qa" # 벡터를 저장할 테이블/컬렉션 이름

# 임베딩 모델 초기화
# Pgvector에 저장될 벡터의 차원과 일치해야 합니다. (all-MiniLM-L6-v2는 384차원)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def check_db_connection():
    """데이터베이스 연결을 테스트하는 함수 (선택 사항)"""
    try:
        conn = psycopg2.connect(PG_CONNECTION_STRING.replace("postgresql+psycopg2://", "postgresql://"))
        cursor = conn.cursor()
        cursor.execute("SELECT 1;")
        print("[INFO] Successfully connected to PostgreSQL.")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"[ERROR] Could not connect to PostgreSQL: {e}")
        print("Please ensure PostgreSQL is running and PG_CONNECTION_STRING in .env is correct.")
        return False

def ingest_documents():
    if not check_db_connection():
        return

    print(f"Loading documents from {DATA_DIRECTORY}...")
    documents = []
    for filename in os.listdir(DATA_DIRECTORY):
        if filename.endswith(".txt"):
            file_path = os.path.join(DATA_DIRECTORY, filename)
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())
    print(f"Loaded {len(documents)} documents.")

    if not documents:
        print(f"No text documents found in '{DATA_DIRECTORY}'. Please add .txt files to this directory.")
        return

    # 문서 분할 (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(splits)} chunks.")

    # Pgvector에 문서 임베딩 및 저장
    print(f"[INFO] Connecting to PGVector and ingesting documents...")
    try:
        # PGVector 인스턴스 생성 또는 기존 인스턴스 연결
        # collection_name은 PostgreSQL 테이블 이름이 됩니다.
        vectorstore = PGVector.from_documents(
            documents=splits,
            embedding=embeddings,
            connection_string=PG_CONNECTION_STRING,
            collection_name=COLLECTION_NAME,
            pre_delete_collection=True # 기존 컬렉션을 삭제하고 새로 만든다면 활성화
        )
        print(f"[SUCCESS] Documents ingested into PGVector collection '{COLLECTION_NAME}'.")

        # 인덱스 생성을 위해 DB 클라이언트를 사용해야 합니다.
        print(f"[INFO] Please manually create an index on the 'embedding' column for performance.")
        print(f"Example SQL (for ivfflat, assuming 384 dimensions):")
        print(f"CREATE INDEX ON langchain.{COLLECTION_NAME} USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);")
        print(f"Or for HNSW:")
        print(f"CREATE INDEX ON langchain.{COLLECTION_NAME} USING hnsw (embedding vector_l2_ops);")


    except Exception as e:
        print(f"[ERROR] Failed to ingest documents into PGVector: {e}")
        print(f"Ensure pgvector extension is enabled in your PostgreSQL database.")
        print(f"SQL: CREATE EXTENSION IF NOT EXISTS vector;")


if __name__ == "__main__":
    check_db_connection()

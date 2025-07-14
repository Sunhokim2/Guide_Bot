import os
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain_aws import ChatBedrock, BedrockEmbeddings # Bedrock 통합 임포트
from langchain_chroma import Chroma # <-- 이 줄로 변경
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# .env 파일 로드
load_dotenv()

app = FastAPI()

# Jinja2 템플릿 디렉토리 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates_dir = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=templates_dir)

# 환경 변수에서 AWS 리전 로드
AWS_REGION_NAME = os.getenv("AWS_REGION_NAME") # Bedrock 모델이 지원되는 리전으로 변경

# LangChain 모델 및 임베딩 초기화
# ChatBedrock: 사용할 Bedrock 채팅 모델 ID 지정
# https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html#model-ids-text-generation
llm = ChatBedrock(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    region_name=AWS_REGION_NAME,
    streaming=True,
    # Claude 3 모델의 경우 추가 설정 (messages API)
    model_kwargs={"temperature": 0.1, "max_tokens": 1000}
)
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0", # ⭐ ingest_data.py와 동일한 임베딩 모델 ID 사용 ⭐
    region_name=AWS_REGION_NAME
)

# ChromaDB 로드
# ingest_data.py를 통해 미리 데이터가 임베딩되어 있어야 합니다.
persist_directory = "chroma_db"
try:
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = vectorstore.as_retriever()
    print("ChromaDB loaded successfully.")
except Exception as e:
    print(f"Error loading ChromaDB: {e}. Please run ingest_data.py first to create the database.")
    retriever = None # DB 로드 실패 시 retriever를 None으로 설정

# RAG 프롬프트 정의
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 친절하고 유용한 챗봇입니다.
다음 검색된 문서를 사용하여 사용자 질문에 **Markdown 형식**으로 답변하세요.
답변은 읽기 쉽고 명확해야 하며, 필요한 경우 **목록(리스트)**, **굵은 글씨**, **줄바꿈** 등을 적극적으로 활용하세요.

제공된 컨텍스트:
{context}"""),
    ("human", "{input}")
])

# 문서 체인 및 검색 체인 설정
if retriever:
    document_chain = create_stuff_documents_chain(llm, rag_prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
else:
    # DB 로드 실패 시, RAG 없이 기본 LLM 응답 체인
    print("RAG retriever is not available. Falling back to direct LLM chat.")
    direct_chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 친절한 챗봇입니다."),
        ("human", "{input}")
    ])
    retrieval_chain = direct_chat_prompt | llm | StrOutputParser()


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    루트 경로 요청 시 Jinja2 템플릿을 렌더링하여 챗봇 UI를 제공합니다.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    웹소켓 연결을 처리하고, 클라이언트 메시지를 RAG 체인으로 전달하여
    스트리밍 응답을 실시간으로 클라이언트에 중계합니다.
    """
    await websocket.accept()
    print("WebSocket connection accepted.")
    try:
        while True:
            user_message = await websocket.receive_text()
            print(f"Received message from client: {user_message}")

            # LangChain RAG 체인 호출 및 스트리밍 응답
            full_response = ""
            try:
                # ChatBedrock도 .stream() 메서드를 지원하여 스트리밍 가능
                for chunk in retrieval_chain.stream({"input": user_message}):
                    # LangChain Bedrock 스트리밍 응답 구조 확인 필요.
                    # create_retrieval_chain 사용 시 'answer' 키에 최종 답변이 담김
                    if "answer" in chunk:
                        content_to_send = chunk["answer"]
                    elif "content" in chunk: # 다른 LangChain runnable에서 직접 content가 오는 경우
                        content_to_send = chunk["content"]
                    else:
                        content_to_send = ""

                    if content_to_send:
                        await websocket.send_text(content_to_send)
                        full_response += content_to_send

                print(f"Full response sent: {full_response}")

            except Exception as e:
                print(f"LangChain RAG chain error with Bedrock: {e}")
                await websocket.send_text(f"Error processing your request: {e}")

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
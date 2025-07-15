import os
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# LangChain AWS 관련 라이브러리 임포트
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS, PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# .env 파일 로드
load_dotenv()

app = FastAPI()

# Jinja2 템플릿 디렉토리 설정
templates = Jinja2Templates(directory="templates")

# # TODO AWS에서 작업하기
# # --- AWS Bedrock 및 LangChain 설정 ---
# # 환경 변수에서 AWS 리전 로드
# AWS_REGION_NAME = os.getenv("AWS_REGION_NAME")
# # LangChain 모델 및 임베딩 초기화
# llm = ChatBedrock(
#     model_id="anthropic.claude-3-haiku-20240307-v1:0",
#     region_name=AWS_REGION_NAME, # region_name 직접 전달
#     streaming=True,
#     model_kwargs={"temperature": 0.1, "max_tokens": 1000}
# )
# embeddings = BedrockEmbeddings(
#     model_id="amazon.titan-embed-text-v2:0",
#     region_name=AWS_REGION_NAME # region_name 직접 전달
# )

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.1,
    max_tokens=1000,
    streaming=True,
    top_p=0.9
)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# pgvector 로컬 로드
PG_CONNECTION_STRING = os.getenv("PG_CONNECTION_STRING")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
try:
    vectorstore = PGVector(
        connection_string=PG_CONNECTION_STRING,
        embedding_function=embeddings,
        collection_name="guide_bot_documents"  # ingest_data_local.py에서 사용한 컬렉션 이름과 동일하게
    )
    # Retriever 설정
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("PGVector VectorStore connected and ready.")

except Exception as e:
    print(f"[ERROR] Failed to connect to PGVector or retrieve data: {e}")
    print(
        "Please ensure your PostgreSQL database is running, pgvector extension is enabled, and PG_CONNECTION_STRING is correct.")
    retriever = None

# RAG 체인 구조 개선 (안정적인 스트리밍을 위해)
if retriever:
    LMS_SERVICE_NAME = "lgcms"

    template = """당신은 {LMS_SERVICE_NAME}의 친절하고 유용한 학습 가이드 챗봇입니다.
사용자의 {LMS_SERVICE_NAME} 관련 질문에 대해 다음 '제공된 컨텍스트'를 바탕으로 친절하고 정확하게 답변해 주세요.

답변은 다음 지침을 따르세요:
- **Markdown 형식**으로 작성하여 읽기 쉽고 명확하게 표현하세요. 필요하다면 **목록, 굵은 글씨, 줄바꿈** 등을 적극적으로 활용하세요.
- '제공된 컨텍스트' 내에 URL 정보가 포함되어 있고, 그 URL이 답변 내용과 직접적으로 관련이 있다면, 답변 마지막에 해당 URL을 **하나만** 명확하게 제공해주세요.
- URL은 `[관련 페이지 바로가기](URL)` 와 같은 Markdown 링크 형식으로 제공해주세요.
- 만약 컨텍스트에 관련 URL이 없거나, 답변과 직접적인 관련이 없다면 URL을 제공하지 마세요.
- URL이 여러 개 관련될 경우, 가장 핵심적인 URL 하나만 선택하여 제공하세요.

제공된 컨텍스트:
{{context}}

질문: {{question}}

답변:"""
    prompt = ChatPromptTemplate.from_template(template)


    def format_docs(docs):
        """검색된 문서들을 하나의 문자열로 합칩니다."""
        return "\n\n".join(doc.page_content for doc in docs)


    # RAG 체인을 구성합니다.
    # RunnablePassthrough를 사용하여 사용자 질문을 체인 전체에 전달합니다.
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}  # 초기 입력: {"question": user_message}
            | RunnablePassthrough.assign(
        context=lambda x: format_docs(x["context"]),  # 검색된 문서를 포맷
        LMS_SERVICE_NAME=lambda x: LMS_SERVICE_NAME  # LMS_SERVICE_NAME 값 추가
    )
            | prompt
            | llm
            | StrOutputParser()
    )
else:
    # DB 로드 실패 시, RAG 없는 기본 LLM 체인
    print("RAG retriever is not available. Falling back to direct LLM chat.")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 친절한 챗봇입니다."),
        ("human", "{question}")
    ])
    rag_chain = prompt | llm | StrOutputParser()


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """루트 경로 요청 시 Jinja2 템플릿을 렌더링합니다."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """웹소켓 연결 및 RAG 체인을 통한 실시간 스트리밍 응답을 처리합니다."""
    await websocket.accept()
    print("WebSocket connection accepted.")
    try:
        while True:
            user_message = await websocket.receive_text()
            print(f"Received message from client: {user_message}")

            full_response = ""
            try:
                # 개선된 RAG 체인을 스트리밍으로 호출합니다.
                # 이제 stream()은 문자열 토큰을 직접 생성합니다.
                for chunk in rag_chain.stream(user_message):
                    await websocket.send_text(chunk)
                    full_response += chunk

                # 스트림이 모두 끝나면, 프론트엔드에 종료 신호를 보냅니다.
                await websocket.send_text("__END_OF_STREAM__")
                print(f"Full response sent, followed by end-of-stream signal.")

            except Exception as e:
                print(f"Error during RAG chain execution: {e}")
                error_message = f"오류가 발생했습니다: {e}"
                await websocket.send_text(error_message)
                await websocket.send_text("__END_OF_STREAM__")  # 에러 발생 시에도 UI 활성화를 위해 신호 전송

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"An unexpected error occurred in websocket: {e}")
    finally:
        # 연결이 어떤 이유로든 끊어지면 정리합니다.
        print("Closing WebSocket connection.")


if __name__ == "__main__":
    import uvicorn

    # host를 "0.0.0.0"으로 설정하여 외부에서도 접속 가능하게 합니다.
    uvicorn.run(app, host="0.0.0.0", port=8000)

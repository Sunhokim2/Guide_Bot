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
    model="gpt-4o",
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

    template = """# Role : 서비스 가이드 챗봇
당신은 {LMS_SERVICE_NAME}의 친절하고 유용한 가이드 챗봇입니다.
질문자의 물음을 기반으로 답변을 한다. 간결하게 답변하는 챗봇입니다.

## Iteration Process:
지속적인 학습 및 개선: 제공된 컨텍스트와 사용자의 질문을 통해 얻은 정보를 바탕으로 답변의 정확성과 유용성을 지속적으로 개선하려 노력합니다.
피드백 반영: 만약 답변이 부족하거나 불명확할 경우, 사용자의 추가 질문이나 명시적인 피드백을 통해 부족한 부분을 인식하고 다음 답변에 반영하려 합니다.

## Operating principles:
컨텍스트 기반 답변:제공된 컨텍스트(사용자 질문과 관련된 서비스 문서 내용)만을 사용하여 질문에 답변합니다.
    정보 부족 시 대응: 만약 컨텍스트에 답변에 필요한 정보가 없다면, "죄송합니다, 현재 제가 가진 정보로는 해당 질문에 대한 답변을 드릴 수 없습니다. 다른 질문이 있으시면 언제든지 말씀해주세요."와 같이 정중하게 답변 불가를 알리세요. 절대 없는 정보를 지어내거나 추측해서 답변하지 마세요.
정확성 및 신뢰성: 답변하는 모든 정보는 사실에 기반하며, {LMS_SERVICE_NAME} 서비스에 대한 최신 정보를 반영해야 합니다.
사용자 중심: 사용자의 의도를 정확히 파악하고, 그들의 문제 해결에 가장 도움이 되는 방식으로 정보를 제공하는 데 집중합니다.
답변의 정확성: 제공된 컨텍스트 외의 내용에 대해 답변한 것을 사용자가 신뢰할 수 있는지 물어보면 "이는 채팅봇의 개인적인 의견으로 100% 신뢰할 순 없으니 주의해주세요!" 라고 답변한다.

## Workflow Guidelines
질문 이해 및 컨텍스트 분석:
    사용자의 질문을 정확히 이해하고, 제공된 컨텍스트에서 질문과 가장 관련성 높은 정보를 식별합니다.
간결하고 명확한 답변 생성:
    핵심 정보 우선: 사용자가 쉽게 이해할 수 있도록 간결하고 명확한 문장을 사용합니다. 불필요한 서론이나 반복은 피하고, 핵심 정보를 바로 전달합니다.
    친절하고 전문적인 어조: 항상 친절하고 공손하며 전문적인 태도를 유지합니다.
    Markdown 형식 활용: 답변의 가독성을 높이기 위해 필요하다면 굵은 글씨, 기울임 글씨, `코드 블록`, 목록(`-` 또는 `1.`), 그리고 줄바꿈(`\n\n`)을 적극적으로 활용합니다.
    사용자 질문 반복 피하기: 답변 시작 시 사용자 질문을 반복하지 말고, 답변으로 바로 들어갑니다.
URL 제공 규칙:
    제공 조건: 오직 사용자 질문에 대한 직접적인 답변을 완료한 후, '제공된 컨텍스트' 내에 있는 URL 정보가 답변 내용과 매우 밀접하게 관련되어 추가적인 정보나 행동이 필요하다고 판단될 때만 URL을 제공합니다.
    형식 및 위치: URL은 답변의 본문이 끝난 후 가장 마지막에 `[자세히 알아보기](URL)` 또는 `[관련 페이지 바로가기](URL)` 와 같은 Markdown 링크 형식으로 하나만 명확하게 제공합니다.
    미제공 조건: 만약 컨텍스트에 관련 URL이 없거나, 답변과 직접적인 관련성이 낮거나, 이미 답변으로 충분하여 추가 URL이 불필요하다고 판단되면 URL을 제공하지 않습니다.
    다중 URL 처리: 여러 관련 URL이 컨텍스트에 있다면, 사용자에게 가장 유용하고 핵심적인 URL 하나만 신중하게 선택하여 제공합니다.


## Examples
질문 : 여기서 강의 다운로드를 하려면요?
컨텍스트 : (RAG를 통해 검색된 관련 문서 내용)
LGCMS에서는 수강 중인 강의의 자료를 다운로드할 수 있습니다. 강의실 입장 후 '강의 자료' 탭에서 원하는 파일을 클릭하여 다운로드하세요. 파일 형식은 PDF, PPT, Word 등 다양합니다.
기대하는 답변 : 강의를 다운로드 하는 건 할 수 없습니다. 혹시, 수강 중인 강의의 자료를 다운 받으시려고 하나요? 먼저 수강 중인 강의실로 입장하고. 그 후 강의 자료 탭을 클릭하시면 원하시는 파일을 다운로드 할 수 있습니다. PDF, PPT, Word 등 다양한 형식의 자료를 이용하실 수 있습니다.


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
            {"context": retriever, "question": RunnablePassthrough()}
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

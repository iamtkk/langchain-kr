import os
import streamlit as st
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_openai import AzureChatOpenAI
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import load_prompt
from langchain import hub
from langchain_teddynote import logging
from dotenv import load_dotenv

load_dotenv(override=True)

# 캐시 디렉토리 생성
if not os.path.exists("./.cache"):
    os.mkdir(".cache")

if not os.path.exists("./.cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists("./.cache/embeddings"):
    os.mkdir(".cache/embeddings")

logging.langsmith("iamtk-streamlit")

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), 
    "https://cognitiveservices.azure.com/.default"
)

llm = AzureChatOpenAI(
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_ad_token_provider=token_provider,
)


st.title("PDF 기반 QA")

# 대화기록을 저장하기 위한 용도로 생성한다.
if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")
    # 파일 업로드
    uploaded_file = st.file_uploader("PDF 파일 업로드", type=["pdf"])

    selected_prompt = "prompts/pdf-rag.yaml"

def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

# 파일이 업로드 되었을 때
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

if uploaded_file:
    embed_file(uploaded_file)

# 챗봇 체인 생성
def create_chain(prompt_filepath):
    # prompt 적용
    prompt = load_prompt(prompt_filepath, encoding="utf-8")

    output_parser = StrOutputParser()

    return prompt | llm | output_parser

if clear_btn:
    st.session_state["messages"] = []

print_messages()

user_input = st.chat_input("궁금한 내용을 물어보세요")

if user_input:
    # 사용자 입력 출력
    st.chat_message("user").write(user_input)

    # 챗봇 체인 생성
    chain = create_chain(selected_prompt)

    # prompt_type이 "요약"일 때는 "ARTICLE" 키를 사용
    input_key = "ARTICLE" if selected_prompt == "요약" else "question"
    response = chain.stream({input_key: user_input})

    with st.chat_message("assistant"):
        container = st.empty()
        ai_answer = "" 
        for chunk in response:
            ai_answer += chunk
            container.markdown(ai_answer)

    # 챗봇 체인 실행
    # ai_answer = chain.invoke({"question": user_input})

    # 챗봇 체인 출력
    # st.chat_message("assistant").write(ai_answer)
    
    # 대화기록 추가
    add_message("user", user_input)
    add_message("assistant", ai_answer)
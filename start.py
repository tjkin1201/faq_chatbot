# 필요한 모듈 불러오기
import os
import json
import streamlit as st
from typing import List, Any, Tuple
import requests
from requests.exceptions import ConnectionError
from dotenv import load_dotenv
from streamlit.runtime.uploaded_file_manager import UploadedFile

from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader

# 페이지 설정을 가장 먼저 해야 함
st.set_page_config(
    page_title="청약 FAQ 챗봇",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

"""
PDF 문서 기반 질의응답 시스템
"""

# 환경변수 로드
load_dotenv()

# 상수 정의
OLLAMA_MODELS = {
    "Llama 3.2 (최신 추천)": "llama3.2",
    "Qwen2.5 (중국어/한국어 우수)": "qwen2.5:7b",
    "Yi-1.5 (다국어 성능 우수)": "yi:34b",
    "DeepSeek Coder (코딩 특화)": "deepseek-coder:6.7b",
    "Mistral (한국어 가능)": "mistral",
    "Llama2 (한국어 가능)": "llama2",
    "Gemma (한국어 가능)": "gemma:2b",
    "CodeLlama (코딩 특화)": "codellama:7b",
    "Phi-3 (Microsoft 최신)": "phi3:mini"
}

OPENAI_MODELS = {
    "GPT-4o (최신 추천)": "gpt-4o",
    "GPT-4 Turbo": "gpt-4-turbo",
    "GPT-4": "gpt-4",
    "GPT-3.5 Turbo": "gpt-3.5-turbo"
}

EMBEDDING_MODELS = {
    "BGE-M3 (다국어 최고성능)": "BAAI/bge-m3",
    "BGE-Large (한국어 우수)": "BAAI/bge-large-en-v1.5",
    "E5-Large-V2 (최신 추천)": "intfloat/e5-large-v2",
    "Multilingual-E5-Large": "intfloat/multilingual-e5-large",
    "Ko-SBERT-multitask": "jhgan/ko-sbert-multitask",
    "Ko-SBERT-nli (한국어 특화)": "jhgan/ko-sbert-nli",
    "Jina-v2-Korean": "jinaai/jina-embeddings-v2-base-ko",
    "KoSimCSE-RoBERTa": "BM-K/KoSimCSE-roberta-multitask",
    "All-MPNet-Base-v2": "sentence-transformers/all-mpnet-base-v2",
    "Paraphrase-Multilingual": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "MiniLM-L6-v2 (기본)": "sentence-transformers/all-MiniLM-L6-v2"
}


def initialize_embeddings(model_name: str, use_cuda: bool) -> HuggingFaceEmbeddings:
    """임베딩 모델 초기화"""
    try:
        # meta tensor 문제를 피하기 위해 CPU로 먼저 초기화
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODELS[model_name],
            model_kwargs={'device': 'cpu'}  # 항상 CPU로 초기화
        )
        if use_cuda:
            st.info("GPU 사용이 선택되었지만, 호환성을 위해 CPU를 사용합니다.")
        return embeddings
    except Exception as e:
        st.error(f"임베딩 모델 초기화 중 오류 발생: {str(e)}")
        st.error("기본 임베딩 모델로 다시 시도합니다.")
        try:
            # 기본 모델로 fallback
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            return embeddings
        except Exception as fallback_error:
            st.error(f"기본 모델도 실패했습니다: {str(fallback_error)}")
            st.stop()


def check_ollama_status() -> Tuple[bool, str]:
    """Ollama 서비스 상태 확인"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            installed_models = [
                model["name"] for model in response.json()["models"]
            ]
            return True, f"설치된 모델: {', '.join(installed_models)}"
        return False, "Ollama 서비스 응답 오류"
    except ConnectionError:
        return False, "Ollama 서비스가 실행되지 않았습니다"
    except Exception as e:
        return False, f"알 수 없는 오류: {str(e)}"


def check_model_installed(model_name: str) -> bool:
    """Ollama 모델이 설치되어 있는지 확인"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            installed_models = []
            for model in response.json()["models"]:
                installed_models.append(model["name"])
            return model_name in installed_models
        return False
    except Exception:
        return False


def install_model(model_name: str) -> bool:
    """Ollama 모델 설치"""
    try:
        # 스트림으로 모델 설치 진행 상태 받기
        response = requests.post(
            "http://localhost:11434/api/pull",
            json={"name": model_name},
            stream=True
        )
        
        if response.status_code != 200:
            st.error(f"모델 설치 중 오류 발생: HTTP {response.status_code}")
            return False

        for line in response.iter_lines():
            if line:
                status = json.loads(line.decode())
                if 'status' in status:
                    st.write(f"진행 중: {status['status']}")
                if 'error' in status:
                    st.error(f"설치 오류: {status['error']}")
                    return False
        return True
    except requests.exceptions.ConnectionError:
        st.error("Ollama 서비스에 연결할 수 없습니다.")
        return False
    except Exception as e:
        st.error(f"예상치 못한 오류 발생: {str(e)}")
        return False


def get_chat_model(model_type: str, model_name: str) -> Any:
    """채팅 모델 초기화"""
    if model_type == "Ollama":
        # Ollama 서비스 상태 확인
        status, message = check_ollama_status()
        if not status:
            st.error(f"""
            Ollama 연결 오류: {message}
            
            해결 방법:
            1. WSL2에서 다음 명령어로 Ollama 서비스를 시작하세요:
               `sudo systemctl start ollama`
               또는
               `ollama serve`
            
            2. Ollama 서비스가 정상적으로 실행 중인지 확인하세요:
               `curl http://localhost:11434/api/tags`
            """)
            st.stop()

        selected_model = OLLAMA_MODELS[model_name]
        
        # 모델이 설치되어 있는지 확인
        if not check_model_installed(selected_model):
            with st.spinner(f"{selected_model} 모델을 다운로드하고 있습니다..."):
                if not install_model(selected_model):
                    st.error(f"""
                    {selected_model} 모델 설치 실패!
                    
                    수동으로 설치해주세요:
                    `ollama pull {selected_model}`
                    """)
                    st.stop()
                else:
                    st.success(f"{selected_model} 모델이 설치되었습니다!")

        return ChatOllama(
            model=selected_model,
            temperature=0,
            base_url="http://localhost:11434",
            timeout=120
        )
    else:  # OpenAI
        return ChatOpenAI(
            model=OPENAI_MODELS[model_name],
            temperature=0
        )


def save_uploadedfile(uploadedfile: UploadedFile) -> str:
    """임시폴더에 파일 저장"""
    temp_dir = "PDF_임시폴더"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.read())
    return file_path


def pdf_to_documents(pdf_path: str) -> List[Document]:
    """저장된 PDF 파일을 Document로 변환"""
    documents = []
    loader = PyMuPDFLoader(pdf_path)
    doc = loader.load()
    for d in doc:
        d.metadata['file_path'] = pdf_path
    documents.extend(doc)
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Document를 더 작은 document로 변환"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,  # 토큰 기준 300~512 사이 권장
        chunk_overlap=40,  # 토큰 기준 20~50 사이 권장
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def save_to_vector_store(documents: List[Document]) -> None:
    """Document를 벡터DB로 저장"""
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_rag_chain(model_type: str, model_name: str) -> Runnable:
    """RAG 체인 생성"""
    template = """
    당신은 청약 제도와 부동산 정책에 대해 전문적인 지식을 갖춘 상담사입니다.
    주어진 컨텍스트를 기반으로 질문에 답변해주세요.
    
    답변 형식을 반드시 다음과 같이 작성해주세요:

    1. 핵심 답변: 
    - 질문에 대한 명확한 "예/아니오" 또는 핵심 결론을 1문장으로 제시
    
    2. 상세 설명:
    - 관련 법규나 규정 인용
    - 구체적인 사례나 조건 설명
    - 예외 사항이 있다면 언급
    
    3. 관련 주의사항:
    - 실수하기 쉬운 부분 안내
    - 유의해야 할 법적/제도적 사항
    - 추가 확인이 필요한 사항
    
    4. 예시:
    - 질문과 유사한 사례를 포함하여 답변을 풍부하게 만드세요
    
    지침:
    - 항상 정확하고 객관적인 정보만 제공하세요
    - 불확실한 내용은 제외하세요
    - 전문 용어는 반드시 쉬운 설명을 덧붙이세요
    - 답변은 친절하고 공손한 어투를 사용하세요
    - 컨텍스트에 없는 내용은 언급하지 마세요
    
    컨텍스트:
    {context}

    질문: {question}

    답변: """

    custom_rag_prompt = PromptTemplate.from_template(template)
    
    # 모델 초기화 시 temperature 조정
    if model_type == "Ollama":
        model = ChatOllama(
            model=OLLAMA_MODELS[model_name],
            temperature=0.1,  # 더 정확한 답변을 위해 낮춤
            base_url="http://localhost:11434",
            timeout=120
        )
    else:  # OpenAI
        model = ChatOpenAI(
            model=OPENAI_MODELS[model_name],
            temperature=0
        )

    return custom_rag_prompt | model | StrOutputParser()


def display_reference_docs(docs: List[Document]) -> None:
    """참고 문서의 위치를 표시"""
    with st.expander("참고한 문서 보기"):
        for i, doc in enumerate(docs, 1):
            page_info = doc.metadata.get('page_number', '페이지 정보 없음')
            title_info = doc.metadata.get('title', '목차 정보 없음')
            st.markdown(f"**문서 {i}:**")
            if title_info != '목차 정보 없음':
                st.write(f"목차: {title_info}")
            if page_info != '페이지 정보 없음':
                st.write(f"페이지: {page_info}")
            st.write(doc.page_content)

# 문서 목차 예시
# 문서 3:
# 목차:
# 전매제한 관련 Q&A
# Q426. 주택의 전매제한이 무엇인가요?
# Q427. 전매제한을 적용받는 대상과 사업주체의 의미는 무엇인가요?
# Q428. 30세대 미만의 주택을 공급하려는 경우에도 전매제한을 적용받는지?
# Q429. 30세대 이상 건설하지만 「주택법 시행령」 제27조제4항에 해당하여
# 「주택법」 제15조에 따른 사업계획승인을 받지 않고 건축허가만으로
# 주택 외의 시설과 주택을 동일 건축물로 건축(주상복합)하는 경우
# 전매제한을 적용받는지?


def postprocess_response(response: str) -> str:
    """모델의 답변을 후처리하여 품질을 향상"""
    # 불필요한 공백 제거
    response = response.strip()
    
    # 문법 및 어투 개선 (예: 한국어 맞춤법 검사 라이브러리 활용 가능)
    # response = korean_spell_checker(response)
    
    # 답변 형식 검증 및 수정
    if not response.endswith("."):
        response += "."
    
    return response


@st.cache_data
def process_question(user_question: str, model_type: str, model_name: str):
    """사용자 질문에 대한 RAG 처리"""
    # 벡터 DB 호출
    new_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # 관련 문서 3개를 호출하는 Retriever 생성
    retriever = new_db.as_retriever(search_kwargs={"k": 3})
    # 사용자 질문을 기반으로 관련문서 3개 검색
    retrieve_docs: List[Document] = retriever.invoke(user_question)

    # RAG 체인 선언
    chain = get_rag_chain(model_type, model_name)
    # 질문과 문맥을 넣어서 체인 결과 호출
    response = chain.invoke({
        "question": user_question,
        "context": retrieve_docs
    })

    return response, retrieve_docs


def check_api_key() -> bool:
    """OpenAI API 키 확인"""
    api_key = os.getenv("OPENAI_API_KEY")
    return bool(api_key)


def main():
    """메인 함수"""
    st.title("PDF 문서 기반 질의응답 시스템")

    # 사이드바에 모델 설정 추가
    with st.sidebar:
        st.header("모델 설정")
        
        # CUDA 사용 여부 선택
        use_cuda = st.checkbox(
            "CUDA 사용 (GPU)",
            value=False,
            help="NVIDIA RTX 4090과 같은 GPU를 활용하려면 선택하세요."
        )
        
        # 임베딩 모델 선택
        embedding_model = st.selectbox(
            "임베딩 모델 선택",
            list(EMBEDDING_MODELS.keys()),
            help="문서 분석에 사용할 임베딩 모델을 선택하세요. BGE-M3는 최신 다국어 모델로 성능이 우수합니다."
        )
        
        # 전역 변수 업데이트
        global embeddings
        embeddings = initialize_embeddings(embedding_model, use_cuda)
        
        # Ollama 서비스 상태 확인
        ollama_status, ollama_message = check_ollama_status()
        
        if ollama_status:
            st.success("Ollama 서비스가 실행 중입니다")
            st.info(ollama_message)
            default_model = "Ollama"
        else:
            st.warning("""
            Ollama 서비스가 실행되지 않았습니다.
            OpenAI를 사용하거나, WSL2에서 Ollama를 설정해주세요.
            
            설치 방법:
            1. `curl https://ollama.ai/install.sh | sh`
            2. `systemctl --user start ollama`
            3. `ollama pull mistral`
            """)
            default_model = "OpenAI"
        
        model_type = st.radio(
            "모델 종류 선택",
            ["OpenAI", "Ollama"],
            index=0 if default_model == "OpenAI" else 1,
            help="OpenAI는 API 키가 필요하며, Ollama는 로컬에서 실행되는 무료 모델입니다."
        )

        if model_type == "Ollama":
            if not ollama_status:
                st.error("Ollama 서비스를 먼저 실행해주세요.")
                st.stop()
            
            model_name = st.selectbox(
                "Ollama 모델 선택",
                list(OLLAMA_MODELS.keys()),
                help="한국어 처리에 최적화된 모델들입니다."
            )
        else:  # OpenAI
            if not check_api_key():
                st.error("OpenAI API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
                return
            
            model_name = st.selectbox(
                "OpenAI 모델 선택",
                list(OPENAI_MODELS.keys())
            )

    # 메인 영역
    uploaded_file = st.file_uploader(
        "PDF 파일을 업로드 해주세요",
        type=['pdf']
    )

    if uploaded_file is not None:
        file_path = save_uploadedfile(uploaded_file)
        documents = pdf_to_documents(file_path)
        smaller_documents = chunk_documents(documents)
        save_to_vector_store(smaller_documents)
        st.success("PDF 파일이 성공적으로 처리되었습니다!")

        user_question = st.text_input("질문을 입력하세요")
        
        if user_question:
            with st.spinner('답변을 생성하고 있습니다...'):
                response, docs = process_question(
                    user_question, model_type, model_name
                )
                st.write(response)
                
                # 참고한 문서 보여주기
                display_reference_docs(docs)


if __name__ == "__main__":
    main()

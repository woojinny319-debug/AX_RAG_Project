import os

os.environ["OPENAI_API_KEY"] = "open ai api key"  # 실제 API 키로 교체

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

def build_and_run_rag():
    print("1. K-IFRS 기준서 PDF 로딩 중...")
    try:
        loader = PyMuPDFLoader("k_ifrs_1115.pdf")
        docs = loader.load()
    except Exception as e:
        print(f"PDF 로딩 에러: {e}. 파일 경로를 확인하세요.")
        return

    print("2. 문서 분할(Chunking) 진행 중...")
    # 글자 수 기준이되, 문단이 최대한 깨지지 않게 구분자(separators) 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    splits = text_splitter.split_documents(docs)

    print("3. Vector DB 구축 및 임베딩 진행 중...")
    # 로컬 폴더에 ChromaDB 저장
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory="./chroma_db"
    )
    # 검색기 세팅: 가장 유사도 높은 문서 3개 가져오기
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print("4. 프롬프트 및 LLM 체인 세팅 중...")
    system_prompt = (
        "당신은 공인회계사를 돕는 전문 AI 어시스턴트입니다.\n"
        "아래 제공된 [K-IFRS 기준서 내용]에만 기반하여 사용자의 질문에 답하십시오.\n"
        "주어진 내용에 답이 없다면, 절대 유추하지 말고 '제공된 기준서 내용에서는 해당 답을 찾을 수 없습니다'라고 명확히 말하십시오.\n"
        "답변 시 반드시 참조한 문단의 내용을 간략히 언급하여 출처를 밝히십시오.\n\n"
        "[K-IFRS 기준서 내용]:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 비용 효율성을 위해 gpt-4o-mini 모델 사용, 환각 통제를 위해 temperature 0(창의성 배제) 설정
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) 
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("\n--- RAG 챗봇 준비 완료 ---")
    
    # 5. 테스트 질의
    query = "고객에게 제품을 인도하기 전에 받은 선수금은 언제 수익으로 인식해야 하는가?"
    print(f"\nQ: {query}")
    
    response = rag_chain.invoke({"input": query})
    print(f"\nA: {response['answer']}")

if __name__ == "__main__":
    build_and_run_rag()
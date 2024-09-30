import os
import base64

import faiss
import numpy as np
from langchain.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough, RunnableMap
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
import base64
import json
from typing import TypedDict
import os
import json
import pickle
from langchain_teddynote import logging

load_dotenv()
# 프로젝트 이름을 입력합니다.
logging.langsmith("my_rag")

# GraphState 상태를 저장하는 용도로 사용합니다.
class GraphState(TypedDict):
    filepath: str  # 문서의 파일 경로. 예: "/path/to/file.pdf"
    filetype: str  # 파일의 유형. 보통 "pdf"와 같은 파일 형식을 나타냄
    page_numbers: list[int]  # 문서의 페이지 번호 목록. 예: [1, 2, 3]
    batch_size: int  # 한 번에 처리할 페이지나 데이터의 배치 크기
    split_filepaths: list[str]  # 분할된 PDF 파일 경로 목록. 예: ["/path/to/split1.pdf", "/path/to/split2.pdf"]
    analyzed_files: list[str]  # 분석이 완료된 파일 경로 목록
    page_elements: dict[int, dict[str, list[dict]]]  # 각 페이지에 포함된 요소들. 페이지 번호를 키로 하고, 그 안에 텍스트, 이미지 등의 요소가 딕셔너리로 저장됨
    page_metadata: dict[int, dict]  # 각 페이지의 메타데이터. 페이지 번호를 키로 하고 해당 페이지의 다양한 메타데이터가 저장됨
    page_summary: dict[int, str]  # 각 페이지에 대한 요약 정보. 페이지 번호를 키로 하고 요약된 텍스트가 저장됨
    images: list[str]  # 문서에 포함된 이미지 파일 경로 목록
    images_summary: list[str]  # 각 이미지에 대한 요약 정보 목록
    tables: list[str]  # 문서에 포함된 테이블 정보 목록
    tables_summary: dict[int, str]  # 각 테이블에 대한 요약 정보. 테이블 인덱스를 키로 하고 요약된 텍스트가 저장됨
    texts: list[str]  # 문서에 포함된 텍스트 목록
    texts_summary: list[str]  # 문서의 텍스트에 대한 요약 정보 목록

def load_graph_state_pickle(filepath: str) -> GraphState:
    """Pickle 파일에서 GraphState를 불러옵니다."""
    with open(filepath, 'rb') as f:
        graph_state = pickle.load(f)
    return graph_state


def find_last_page_max_id(graph_state):
    """마지막 페이지의 가장 큰 ID를 찾습니다."""
    last_page = max(graph_state["page_numbers"])
    elements = graph_state["page_elements"].get(last_page, {})
    max_id = -1
    for _, items in elements.items():
        for item in items:
            if "id" in item and item["id"] > max_id:
                max_id = item["id"]
    return max_id

def create_document(id, loaded_state):
    """문서를 생성하는 함수."""
    content = ""
    if id in loaded_state["table_markdown"]:
        content += loaded_state["table_markdown"][id] + "\n"
    if id in loaded_state["table_summary"]:
        content += loaded_state["table_summary"][id] + "\n"
    if id in loaded_state["texts"]:
        content += loaded_state["texts"][id] + "\n"
    if id in loaded_state["text_summary"]:
        content += loaded_state["text_summary"][id] + "\n"
    return Document(page_content=content)

def process_document_for_book(query, book_name, query_embedding, embedding_model,first_id):
    """특정 교재의 인덱스와 피클 파일을 로드하고 문서를 검색."""
    
    # 현재 경로에서 교재의 피클 파일 및 인덱스 파일 경로 설정
    base_name = base64.urlsafe_b64encode(book_name.encode('utf-8')).decode('utf-8')
    path_pickle = os.path.join(os.getcwd(), 'data', 'pickle', f"{base_name}.pkl")
    path_index = os.path.join(os.getcwd(), 'data', 'index', f"{base_name}.index")
    # 피클 파일과 FAISS 인덱스 파일 로드
    if not os.path.exists(path_pickle):
        raise FileNotFoundError(f"FAISS 피클 파일을 찾을 수 없습니다: {path_pickle}")
    loaded_state = load_graph_state_pickle(path_pickle)
    if not os.path.exists(path_index):
        raise FileNotFoundError(f"FAISS 인덱스 파일을 찾을 수 없습니다: {path_index}")
    faiss_index = faiss.read_index(path_index)
    
    # 마지막 페이지에서 가장 높은 ID 찾기
    last_id = find_last_page_max_id(loaded_state)
    print(last_id)
    # first_id=39
    # 문서 리스트 작성
    documents = [create_document(id, loaded_state) for id in range(first_id, last_id)]
    
    # InMemoryDocstore 생성
    docstore = InMemoryDocstore(dict(enumerate(documents, start=first_id)))

    # FAISS 인덱스에서 검색 수행
    # query_embedding = embedding_model.embed_documents([query])[0]
    D, I = faiss_index.search(np.array([query_embedding]).astype(np.float32), faiss_index.ntotal)
    
    # 검색된 IDs를 1차원 배열로 변환
    ids = I.flatten()
    
    # index_to_docstore_id 매핑
    index_to_docstore_id = {int(ids[i]): int(ids[i]) for i in range(len(ids)) if ids[i] != -1}
    # print(index_to_docstore_id)
    # VectorStore 생성
    vectorstore = FAISS(embedding_function=embedding_model, index=faiss_index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)
    
    # 검색기 설정
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'k': 100, "score_threshold": 0.25}
    )

    # 질문과 관련된 문서 추출
    retrieved_docs = retriever.get_relevant_documents(query)
    for i, doc in zip(ids, retrieved_docs):
        print(f"문서 ID: {i}, 내용 요약: {doc.page_content[:10]}")  # 처음 100자만 출력
    print(f"검색된 문서 개수: {len(retrieved_docs)}")
    print(len(retrieved_docs))
    return retrieved_docs

# 메인 코드: 여러 교재를 처리
def 백_inference(query, book_names):
    embedding_model = HuggingFaceEmbeddings(
        model_name='upskyy/bge-m3-korean',
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True},
    )
    
    query_embedding = embedding_model.embed_documents([query])[0]
    
    filtered_docs = []
    seen_contents = set()

    # 여러 교재를 순회하며 처리
    for book_name, first_id in book_names.items():
        print(f"Processing book: {book_name}")
        try:
            retrieved_docs = process_document_for_book(query, book_name, query_embedding, embedding_model,first_id)
            
            # 중복 문서 제거
            for doc in retrieved_docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_contents:
                    filtered_docs.append(doc)
                    seen_contents.add(content_hash)
        except Exception as e:
            print(f"Error processing book '{book_name}': {e}")
    
    print(f"Total unique documents retrieved: {len(filtered_docs)}")
    

    # 13. Ollama 모델 설정
    # model = ChatOllama(model="llama3.1:70b", temperature=0.5)gemma2:27b
    # model = ChatOllama(model="bnksys/yanolja-eeve-korean-instruct-10.8b", temperature=0.5)
    model = ChatOllama(model="ko-gemma-2", temperature=0.85)
    # model = ChatOllama(model="solar-pro", temperature=0.5)
    # model = ChatOllama(model="gemma2:27b", temperature=0.5)

    # 14. 프롬프트 템플릿 설정
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="Answer the question based only on the following context:\n    {context}\n    Question: {question}\n    Format the answer as follows: \"답변 : [answer]\".\n "
    )

    print(len(filtered_docs))
    # 15. 체인 설정
    context_chain = RunnableMap({
        'context': lambda x: "\n".join([str(doc.page_content) for doc in filtered_docs]),  
        'question': RunnablePassthrough()
    })

    # LLMChain을 통해 모델에 프롬프트 전달
    chain = (
        context_chain
        | LLMChain(llm=model, prompt=prompt_template)
    )

    # 결과를 저장할 JSON 파일 경로
    json_file_path = 'results.json'

    # 결과를 \n 문자 기반으로 개행하는 함수
    def format_text(text):
        # \n 문자를 기반으로 개행 처리
        return text.replace("\\n", "\n")

    # 결과를 JSON 파일에 저장하는 함수
    def save_results(results, file_path):
        # 기존 결과가 있으면 읽어오기
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        else:
            data = []

        # 새로운 결과 추가
        data.append(results)

        # 결과를 파일에 저장
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    # 16. 체인 실행
    print(f'Executing RAG chain for query: {query}')

    try:
        
        result = chain.invoke({
            'context': filtered_docs, 
            'question': query
            })
        print(f'Result for query: {result}')
        # output.append(result)
        # 모델 출력에서 답변 부분만 추출
        answer = result['text'].strip()  # 필요시 'text'를 실제 반환 필드명으로 변경
        formatted_answer = format_text(answer)
        # 결과를 JSON 파일에 저장
        save_results({'query': query, 'answer': formatted_answer}, json_file_path)
        
    except Exception as e:
        print(f'Error during RAG chain execution for query: {e}')
        return None
    # 반환된 문서 리스트 반환
    return formatted_answer


book_names = {"견고한데이터엔지니어링":39, "데이터플랫폼설계구축":27}

query = "새로 서비스를 개발하는 신생업체의 경우 클라우드, 온프레미스 어떤것을 추천해?"
answer=백_inference(query, book_names)
print(f"Final Answer: {answer}")

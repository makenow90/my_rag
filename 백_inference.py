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

# 1. 메타데이터 불러오기
def load_metadata(filepath: str):
    with open(filepath, 'rb') as f:
        metadata = pickle.load(f)
    return metadata


def find_metadata_index(doc_id, metadata):
    for index, doc_metadata in enumerate(metadata):
        if doc_metadata.get("source") == doc_id:
            return index
    return None

# 2. 문서 생성 함수 (메타데이터 기반)
def create_document(doc_metadata):
    content = ""
    # 메타데이터의 'type'에 따라 문서 내용을 다르게 구성
    if doc_metadata["type"] == "table":
        content += doc_metadata["content"] + "\n"
    if doc_metadata["type"] == "texts":
        content += doc_metadata["content"] + "\n"
    
    # 추가적으로 다른 유형을 처리할 수 있음 (예: 'image_summary')
    return Document(page_content=content, metadata=doc_metadata)

# 테이블과 텍스트를 분리하는 함수
def extract_table_docs(documents):
    table_docs = []
    texts_docs=[]
    for doc in documents:
        if doc.metadata.get("type") == "table":  # 메타데이터에서 'type'이 'table'인 경우
            table_docs.append(doc)
        elif doc.metadata.get("type") == "texts":  # 메타데이터에서 'type'이 'table'인 경우
            texts_docs.append(doc)
    return table_docs,texts_docs

def process_document_for_book(query, book_name, query_embedding, embedding_model):
    """특정 교재의 인덱스와 피클 파일을 로드하고 문서를 검색."""
    # 현재 경로에서 교재의 피클 파일 및 인덱스 파일 경로 설정
    base_name = base64.urlsafe_b64encode(book_name.encode('utf-8')).decode('utf-8')
    # path_pickle = os.path.join(os.getcwd(), 'data', 'pickle', f"{base_name}.pkl")
    path_index = os.path.join(os.getcwd(), 'data', 'index', f"{base_name}.index")

    if not os.path.exists(path_index):
        raise FileNotFoundError(f"FAISS 인덱스 파일을 찾을 수 없습니다: {path_index}")
    faiss_index = faiss.read_index(path_index)

    # 메타데이터 파일 경로 설정
    metadata_filepath = f"data\\metadata\\{base_name}.pkl"
    metadata = load_metadata(metadata_filepath)

    # 3. 문서 리스트 작성
    documents = [create_document(doc_metadata) for doc_metadata in metadata]
    print(f"총 문서 수: {len(documents)}")

    # 문서 리스트 작성
    # InMemoryDocstore 생성
    docstore = InMemoryDocstore(dict(enumerate(documents)))

    # FAISS 인덱스에서 검색 수행
    # query_embedding = embedding_model.embed_documents([query])[0]
    D, I = faiss_index.search(np.array([query_embedding]).astype(np.float32), faiss_index.ntotal)
    
    # 검색된 IDs를 1차원 배열로 변환
    ids = I.flatten()
    
    
    # index_to_docstore_id 매핑
    index_to_docstore_id = {int(ids[i]): find_metadata_index(int(ids[i]), metadata) for i in range(len(ids)) if ids[i] != -1}
    print(index_to_docstore_id)
    # VectorStore 생성
    vectorstore = FAISS(embedding_function=embedding_model, index=faiss_index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)
    
    # 검색기 설정
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'k': 20, "score_threshold": 0.35}
    )

    # 질문과 관련된 문서 추출
    retrieved_docs = retriever.get_relevant_documents(query)
    
    for i, doc in zip(ids, retrieved_docs):
        print(f"문서 ID: {i}, 내용 요약: {doc.page_content[:30]}")  # 처음 100자만 출력

    print(f"검색된 문서 개수: {len(retrieved_docs)}")
    print(len(retrieved_docs))

    
    # 12. 중복 문서 제거
    seen_contents = set()
    filtered_docs = []
    for doc in retrieved_docs:
        content_hash = hash(doc.page_content)
        if content_hash not in seen_contents:
            filtered_docs.append(doc)
            seen_contents.add(content_hash)

    # 테이블 문서 추출
    table_docs, retrieved_docs = extract_table_docs(retrieved_docs)
    print(f"테이블 제거 문서 개수: {len(retrieved_docs)}")
    print(f"테이블 문서 개수: {len(table_docs)}")
    return retrieved_docs,table_docs


# 메인 코드: 여러 교재를 처리
def 백_inference(query, book_names):
    table_docs = []
    retrieved_docs=[]
    embedding_model = HuggingFaceEmbeddings(
        model_name='upskyy/bge-m3-korean',
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True},
    )
    
    query_embedding = embedding_model.embed_documents([query])[0]

    # 여러 교재를 순회하며 처리
    for book_name in book_names:
        print(f"Processing book: {book_name}")

        docs, tables = process_document_for_book(query, book_name, query_embedding, embedding_model)
        # retrieved_docs와 table_docs에 각각 축적
        retrieved_docs += docs  # 새로운 문서들을 추가
        table_docs += tables  # 새로운 테이블들을 추가
        
    print(f"Total unique documents retrieved: {len(retrieved_docs)}")
    
    print(table_docs)
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

    print(len(retrieved_docs))
    # 15. 체인 설정
    context_chain = RunnableMap({
        'context': lambda x: "\n".join([str(doc.page_content) for doc in retrieved_docs]),  
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
            'context': retrieved_docs, 
            'question': query
            })
        # print(f'Result for query: {result}')
        # output.append(result)
        # 모델 출력에서 답변 부분만 추출
        answer = result['text'].strip()  # 필요시 'text'를 실제 반환 필드명으로 변경

        # for doc in table_docs[:1]:
        #     answer+= "\n"+ doc.page_content
        #     print("\n")
        formatted_answer = format_text(answer)
        print()
        # table_docs가 리스트일 경우에 대한 처리
        if table_docs is not None:
            # 리스트 내 각 항목에 대해 format_text 적용
            table_docs = "\n".join([str(doc.page_content) for doc in table_docs])
                # 리스트가 아닐 경우, 일반 문자열로 처리
            table_docs = format_text(table_docs)
        else:
            table_docs=''

        # print(f'3333333333333{table_docs}')
        # 결과를 JSON 파일에 저장
        save_results({'query': query, 'answer': formatted_answer}, json_file_path)
        
    except Exception as e:
        print(f'Error during RAG chain execution for query: {e}')
        return None
    # 반환된 문서 리스트 반환
    return formatted_answer[:1900], table_docs[:1900]


# book_names = {"견고한데이터엔지니어링":39, "데이터플랫폼설계구축":27, "aws":21}
book_names = {"견고한데이터엔지니어링", "aws"}
query = "단일 퍼블릭 서브넷 VPC 실습 환경 구성에 대한 단계별 가이드"
answer,table_docs=백_inference(query, book_names)
print(f"Final Answer: {answer}")
print(f"Table Docs: {table_docs}")

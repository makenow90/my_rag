import os
import base64

import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough, RunnableMap
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

import base64
import json
from typing import TypedDict
import os
import json
import pickle
from langchain_teddynote import logging
from langchain_community.retrievers import BM25Retriever
from langchain_teddynote.retrievers import (
    KiwiBM25Retriever,
    KkmaBM25Retriever,
    OktBM25Retriever,
)
from langchain_community.document_transformers import LongContextReorder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

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
    # if doc_metadata["type"] == "table":
    #     content += doc_metadata["content"] + "\n"

    # if doc_metadata["type"] == "image":
    #     content += doc_metadata["content"] + "\n"

    if doc_metadata["type"] == "text":
        content += doc_metadata["content"] + "\n"
    
    # 추가적으로 다른 유형을 처리할 수 있음 (예: 'image_summary')
    return Document(page_content=content, metadata=doc_metadata)

# 테이블과 텍스트를 분리하는 함수
def extract_table_docs(documents):
    table_docs = []
    text_docs=[]
    image_docs=[]
    for doc in documents:
        if doc.metadata.get("type") == "table":  # 메타데이터에서 'type'이 'table'인 경우
            table_docs.append(doc.metadata.get("content"))
        elif doc.metadata.get("type") == "image":  # 메타데이터에서 'type'이 'table'인 경우
            image_docs.append(doc.metadata.get("content"))
        elif doc.metadata.get("type") == "text":  # 메타데이터에서 'type'이 'table'인 경우
            text_docs.append(doc)
    return text_docs, table_docs, image_docs

def process_document_for_book(query, book_names, query_embedding, embedding_model):
    sparse_docs = []
    dense_docs = []

    # 1. 여러 교재에서 문서 수집
    for book_name in book_names:
        print(f"Processing book: {book_name}")

        # 교재의 인덱스 및 메타데이터 로드
        base_name = base64.urlsafe_b64encode(book_name.encode('utf-8')).decode('utf-8')
        path_index = os.path.join(os.getcwd(), 'data', 'index', f"{base_name}.index")

        if not os.path.exists(path_index):
            raise FileNotFoundError(f"FAISS 인덱스 파일을 찾을 수 없습니다: {path_index}")
        faiss_index = faiss.read_index(path_index)

        metadata_filepath = f"data\\metadata\\{base_name}.pkl"
        source_docs = load_metadata(metadata_filepath)

        # 문서 리스트 작성
        documents = [create_document(doc_metadata) for doc_metadata in source_docs]
        print(f"총 문서 수: {len(documents)}")

        # InMemoryDocstore 생성
        docstore = InMemoryDocstore(dict(enumerate(documents)))
        print(f'111111111111{len(docstore._dict)}')

        # FAISS 인덱스에서 검색 수행
        D, I = faiss_index.search(np.array([query_embedding]).astype(np.float32), faiss_index.ntotal)
        ids = I.flatten()

        # index_to_docstore_id 매핑
        index_to_docstore_id = {int(ids[i]): find_metadata_index(int(ids[i]), source_docs) for i in range(len(ids)) if ids[i] != -1}

        # VectorStore 생성
        vectorstore = FAISS(embedding_function=embedding_model, index=faiss_index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)

        # 검색기 설정 (리랭커 호출 없이 검색만 수행)
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'k': 40, "score_threshold": 0.30}
        )
        dense_docs.extend(retriever.invoke(query))

        # BM25 검색기 설정 (리랭커 호출 없이 검색만 수행)
        bm25_retriever = KiwiBM25Retriever.from_documents(documents)
        bm25_retriever.k = 40  # 검색 결과 개수 설정
        sparse_docs.extend(bm25_retriever.invoke(query))


    # 2. 수집한 모든 문서 합치기
    retrieved_docs = sparse_docs + dense_docs
    print(f"총 수집된 문서 개수: {len(retrieved_docs)}")

    # 중복 문서 제거
    seen_contents = set()
    filtered_docs = []
    for doc in retrieved_docs:
        content_hash = hash(doc.page_content)
        if content_hash not in seen_contents:
            filtered_docs.append(doc)
            seen_contents.add(content_hash)
    print(f"(중복 제거된) 문서 개수: {len(filtered_docs)}")

    # 3. 리랭킹: 모든 문서에 대해 리랭커를 한 번만 호출
    reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    compressor = CrossEncoderReranker(model=reranker_model, top_n=40)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=None  # 이미 수집된 문서들에 리랭커만 적용
    )

    # 문서 압축 및 리랭킹 수행
    ranked_docs = compression_retriever._get_relevant_documents(
        query,
        collected_docs=filtered_docs
    )
    print(f"리랭커 적용 후 문서 개수: {len(ranked_docs)}")

    # 4. 테이블, 이미지 문서 추출
    text_docs, table_docs, image_docs = extract_table_docs(ranked_docs)
    print(f"텍스트 문서 개수: {len(text_docs)}")
    print(f"테이블 문서 개수: {len(table_docs)}")
    print(f"이미지 문서 개수: {len(image_docs)}")

    # 문서 순서 재조정
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(text_docs)

    return reordered_docs, table_docs, image_docs


# 메인 코드: 여러 교재를 처리
def 백_inference(query, book_names):
    text_docs=[]
    table_docs = []
    image_docs=[]

    embedding_model = HuggingFaceEmbeddings(
        model_name='upskyy/bge-m3-korean',
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True},
    )
    
    query_embedding = embedding_model.embed_documents([query])[0]

    text_docs, table_docs, image_docs = process_document_for_book(query, book_names, query_embedding, embedding_model)

    print(f"Total unique documents retrieved: {len(text_docs)}")
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

    print(len(text_docs))
    # 15. 체인 설정
    context_chain = RunnableMap({
        'context': lambda x: "\n".join([str(doc.page_content) for doc in text_docs]),  
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
            'context': text_docs, 
            'question': query
            })
        # 모델 출력에서 답변 부분만 추출
        answer = result['text'].strip()  # 필요시 'text'를 실제 반환 필드명으로 변경
        formatted_answer = format_text(answer)

        # 결과를 JSON 파일에 저장
        save_results({'query': query, 'answer': formatted_answer}, json_file_path)
        
    except Exception as e:
        print(f'Error during RAG chain execution for query: {e}')
        return None
    # 반환된 문서 리스트 반환
    return formatted_answer[:1900], table_docs, image_docs

book_names = {"견고한데이터엔지니어링", "aws","데이터플랫폼설계구축"}
# book_names = {"aws"}
query = "서브넷 설정 방법 알려줘"
answer, table_docs, image_docs=백_inference(query, book_names)
print(f"Final Answer: {answer}")
print(f"Table Docs: {table_docs}")
print(f"Image Docs: {image_docs}")

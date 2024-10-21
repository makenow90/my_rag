## 코드의 동작 흐름

1. 디스코드에서 질문을 받으면 큐에 적재 (Question_extractor.py)
2. 큐에 적재된 질문을 읽으면서 종류에 따라 추론 코드 불러옴(Consumer.py)
```python
    # Consumer.py의 코드 43~50번줄

    # 백엔드 관련 질문 '백_inference.py' 의 '백_inference'함수 불러옴
    if query.startswith('!백'):
        book_names =  {"aws", "데이터플랫폼설계구축","견고한데이터엔지니어링"}
        query=query.replace('!백','')
        answer,table_answers,image_answers=백_inference(query, book_names)
    elif query.startswith('!운동'):
        book_names =  {"백년운동"}
        query=query.replace('!운동','')
        answer,table_answers,image_answers=운동_inference(query, book_names)
```
3. 데이터나 백엔트 관련된 질문은 해당 추론 실행 (백_inference.py)
4. 운동에 관한 질문은 해당 추론 실행 (운동_inference.py)

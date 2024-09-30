## 코드의 동작 흐름

1. 디스코드에서 질문을 받으면 큐에 적재 (Question_extractor.py)
2. 큐에 적재된 질문을 읽으면서 종류에 따라 추론 코드 불러옴(Consumer.py)
3. 데이터나 백엔트 관련된 질문은 해당 추론 실행 (백_inference.py)
4. 운동에 관한 질문은 해당 추론 실행 (운동_inference.py)

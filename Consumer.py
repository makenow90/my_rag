import pika
import time
import psutil
from datetime import datetime
import threading
import json
from 백_inference import 백_inference
from 운동_inference import 운동_inference


# RabbitMQ 연결 후 기본정보 받음
def connect_to_rabbitmq():
    while True:
        try:
            # 로컬 rabbitmq에 연결
            # BlockingConnection 방식 : 현재 process_message와, consume_messages 함수 두개가 있는데, BlockingConnection 방식을 사용하면
            # 모든 과정이 순차적으로 진행된다. (한번에 하나의 큐만 소비하고 처리해서, langchain 으로 추론을 실행하고 결과물 큐로 보내는 과정)
            # 이렇게 한 이유는 처음부터 GPU 자원의 한계로 langchain을 한번에 한 작업만 실행시키기 위해서 였다.
            connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
            channel = connection.channel()
            # 큐 서버가 꺼졌다 켜져도 큐를 유지하기 위해 durable 옵션을 True로 설정
            channel.queue_declare(queue='in_queue', durable=True, arguments={'x-max-priority': 5})
            return connection, channel
        except pika.exceptions.AMQPConnectionError as e:
            print(f"Failed to connect to RabbitMQ: {e}. Retrying in 5 seconds...")
            time.sleep(5)

# 메시지 처리 함수
def process_message(channel, method, body):
    print(f" body {datetime.now()} : {body.decode()}")

    try:
        message_body = body.decode()
        message_data = json.loads(message_body)

        query = message_data.get('message_content')  # 첫 번째 부분은 query (질문)
        author_info = message_data.get('author_info')  # 나머지 부분은 작성자 정보와 ID
        channel_id = message_data.get('channel_id')   # 채널 ID

    except ValueError:
        print("Invalid message format")
        # 아까 basic_ack=false로 설정했어서, 지금 수동으로 보냄
        channel.basic_ack(delivery_tag=method.delivery_tag)
        return

    image_answers=[]
    if query.startswith('!백'):
        # book_names =  {"견고한데이터엔지니어링", "aws", "데이터플랫폼설계구축"}
        book_names =  {"데이터플랫폼설계구축"}
        query=query.replace('!백','')
        answer,table_answers,image_answers=백_inference(query, book_names)
    elif query.startswith('!운동'):
        book_names =  {"백년운동"}
        query=query.replace('!운동','')
        answer,table_answers,image_answers=운동_inference(query, book_names)
        # table_answers=['c:\\Users\\makenow\\prj\\my_rag\\data\\백년운동\\백년운동\\60.png',
        # 'c:\\Users\\makenow\\prj\\my_rag\\data\\백년운동\\백년운동\\60.png']
        # answer='텍스트  '
    
    print(f"sent query: {query}")

    print(f"sent answer: {answer}")
    if answer is not None:
        print(f"큐 보내기전 답변: {answer}")
        text_data = {"type": "text", "text_answers": answer,'author_info':author_info,'channel_id':channel_id}

        # 처리된 question을 out_queue에 전송
        channel.queue_declare(queue='out_queue', durable=True)  # out_queue 선언
        channel.basic_publish(
            exchange='',
            routing_key='out_queue',
            body=json.dumps(text_data), # 인코딩하여 전송
            properties=pika.BasicProperties(
                delivery_mode=2  # 메시지 내구성 설정
            )
        )
        print(f"Sent '{text_data}' to out_queue")
    
    if table_answers is not None:
        table_data = {"type": "table", "table_answers": table_answers,'author_info':author_info,'channel_id':channel_id}

        # 처리된 question을 out_queue에 전송
        channel.queue_declare(queue='out_queue', durable=True)  # out_queue 선언
        channel.basic_publish(
            exchange='',
            routing_key='out_queue',
            body=json.dumps(table_data),  # 인코딩하여 전송
            properties=pika.BasicProperties(
                delivery_mode=2  # 메시지 내구성 설정
            )
        )
    if image_answers is not None:
        image_data = {"type": "image", "image_answers": image_answers,'author_info':author_info,'channel_id':channel_id}

        # 처리된 question을 out_queue에 전송
        channel.queue_declare(queue='out_queue', durable=True)  # out_queue 선언
        channel.basic_publish(
            exchange='',
            routing_key='out_queue',
            body=json.dumps(image_data),  # 인코딩하여 전송
            properties=pika.BasicProperties(
                delivery_mode=2  # 메시지 내구성 설정
            )
        )

    # 메시지 소비 후 RabbitMQ 서버에 메시지를 처리했음을 알림(ACK, acknowledgment). 따라서 메시지가 큐에서 제거됨
    channel.basic_ack(delivery_tag=method.delivery_tag)

# 메시지 소비 함수 (예외 처리 및 자동 재연결 추가)
def consume_messages(channel):
    # 이전프로그램이 실행 중이 아닐때만 큐를 소비
    while True:
        try:
            # 비동기 메시지 소비 방식인 basic_consume()과 달리, basic_get()은 동기적으로 한 번에 하나의 메시지만 가져옴
            # method_frame: 메시지의 메타데이터(예: delivery_tag)를 포함합니다.
            # body: 큐에서 가져온 메시지의 실제 내용
            # auto_ack=False
            # in_queue를 가져오는 도중에 비정상 종료되거나 멈췄을 때: ACK를 보내지 않았기 때문에, 
            # RabbitMQ는 해당 메시지를 처리되지 않은 것으로 간주하고, 다시 큐에 남겨둠
            # 이를 통해 메시지가 안전하게 처리되었을 때만 큐에서 제거되도록 보장
            method_frame, _, body = channel.basic_get('in_queue', auto_ack=False)
            # 큐에서 데이터 가져온거 성공하면 메세지 처리함
            if method_frame:
                process_message(channel, method_frame, body)
            # else:
            #     print("No messages in queue.")
        # 예외 발생하면 재연결 : RabbitMQ 브로커가 연결을 강제로 닫았을 때, AMQP 프로토콜과 관련된 채널 오류가 발생했을 때, 
        # 네트워크 연결이 끊기거나 문제로 인해 스트림이 중단되었을 때
        except (pika.exceptions.ConnectionClosedByBroker, pika.exceptions.AMQPChannelError, pika.exceptions.StreamLostError) as e:
            print(f"Connection lost: {e}. Reconnecting...")
            connection, channel = connect_to_rabbitmq()  # 재연결
        time.sleep(2)

# 메인스레드 : 큐 연결
connection, channel = connect_to_rabbitmq()

# 종료 플래그 및 스레드 실행
stop_flag = False
try:
    print(' [*] Waiting for messages. To exit press CTRL+C')
    # 서브 스레드를 생성하고, monitor_program_and_consume 함수를 실행
    # 메인 스레드는 사용자 입력(예: 종료 요청)이나 다른 작업을 처리하고, 별도의 스레드에서는 메시지를 계속해서 모니터링하고 소비
    monitor_thread = threading.Thread(target=consume_messages(channel))
    monitor_thread.start()

    # is_alive() 메서드는 모니터링 스레드가 현재 실행 중인지 여부를 반환합니다. 만약 모니터링 스레드가 여전히 실행 중이라면 True를 반환하고, 종료되면 False를 반환
    # join(1) 메서드는 메인 스레드를 모니터링 스레드가 종료될 때까지 대기시킵니다. 인자로 주어진 1은 최대 1초 동안 대기한다는 의미
    # 모니터링 스레드가 계속 실행되는 동안 메인 스레드가 대기하도록 하여, 프로그램이 정상적으로 작동하도록 합니다. 
    # 1초 간격으로 상태를 확인함으로써, CPU 자원을 불필요하게 사용하지 않도록 효율적으로 설계됨
    while monitor_thread.is_alive():
        monitor_thread.join(1)

# 키보드 인터럽트 통한 종료를 가능하게함
except KeyboardInterrupt:
    print("\nGracefully shutting down...")
    stop_flag = True
    connection.close()
    # join()은 스레드 간의 동기화를 보장하여, 모니터링 스레드가 모든 작업을 마치고 종료된 후에 메인 스레드가 종료 메시지를 출력하도록 함
    # join()은 스레드 A가 스레드 B의 작업이 끝날 때까지 기다리게 만듦
    monitor_thread.join()
    print("Shutdown complete.")

# 메인 스레드: RabbitMQ와의 초기 연결, 사용자 입력 대기 및 종료 절차를 처리합니다.
# 서브 스레드: 특정 프로그램의 실행 상태를 모니터링하고, 메시지를 소비하며, 연결 문제를 처리합니다.x
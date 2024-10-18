import discord
from dotenv import load_dotenv
import os
import aio_pika
import asyncio
from datetime import datetime
import json

# 현재 날짜와 시간 가져오기
now = datetime.now()

load_dotenv()
dis_token = os.getenv('dis_token')

# RabbitMQ 메시지 전송 함수 (in_queue로 질문과 작성자 이름, 채널 ID를 전송)
# 비동기 실행을 통해서, 같은 시간에 consume_out_queue() 함수를 통한 답변 수신도 가능하다.
async def send_to_rabbitmq(message_content, message_priority, author_name, channel_id):
    # RabbitMQ 연결
    # await는 비동기 함수(즉, async로 정의된 함수) 내에서 비동기 작업을 기다릴 때 사용하는 키워드입니다. 
    # 비동기 작업이 완료될 때까지 기다렸다가, 그 결과를 받아오는 역할
    connection = await aio_pika.connect_robust("amqp://localhost/")
    async with connection:
        # 현재 채널 
        channel = await connection.channel()  # 채널 생성

        message_body = {"message_content": message_content,'author_info':author_name,'channel_id':channel_id}
        # 기본 큐를 in_queue로 선택하고 메시지를 전송
        await channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(message_body).encode('utf-8'),
                # PERSISTENT 추가로 내구성 있는 메시지로 설정. RabbitMQ가 재시작되더라도 메시지를 잃지 않고 유지
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                # 우선순위 설정
                priority=message_priority
            ),
            routing_key='in_queue'  # in_queue로 전송
        )

        print(f"Sent '{message_content}' with priority {message_priority} and author '{author_name}' to in_queue")

async def consume_out_queue():
    connection = await aio_pika.connect_robust("amqp://localhost/")
    async with connection:
        channel = await connection.channel()
        queue = await channel.declare_queue('out_queue', durable=True)

        # RabbitMQ 큐에서 메시지를 비동기적으로 순회할 수 있는 이터레이터를 생성합니다.
        async with queue.iterator() as queue_iter:
            # 큐에서 하나씩 메시지를 가져오는 비동기 루프
            async for message in queue_iter:
                async with message.process():
                    # 수신한 메시지 파싱
                    message_body = message.body.decode()
                    message_data = json.loads(message_body)

                    message_type = message_data.get("type")
                    channel_id = message_data.get("channel_id")
                    author_info = message_data.get("author_info")
                    
                    # Discord에서 채널 ID를 이용해 채널을 찾음
                    discord_channel = client.get_channel(int(channel_id))
                    
                    if discord_channel:
                        if message_type == "table":
                            # 테이블 데이터(파일 경로 리스트) 처리
                            table_answers = message_data.get("table_answers")
                            print(f"Table answers: {table_answers}")
                            if table_answers:
                                files_to_attach = []
                                for file_path in table_answers:
                                    # 파일이 존재하는지 확인 후 Discord 파일 첨부 리스트에 추가
                                    if os.path.exists(file_path):
                                        files_to_attach.append(discord.File(file_path))
                                    else:
                                        print(f"File not found: {file_path}")
                                        
                                print(files_to_attach)
                                if files_to_attach:
                                    await discord_channel.send(files=files_to_attach)
                                else:
                                    await discord_channel.send(f"No valid table files found. - {author_info}")

                        elif message_type == "image":
                            # 테이블 데이터(파일 경로 리스트) 처리
                            image_answers = message_data.get("image_answers")
                            print(f"image answers: {image_answers}")
                            if image_answers:
                                files_to_attach = []
                                for file_path in image_answers[:9]:
                                    # 파일이 존재하는지 확인 후 Discord 파일 첨부 리스트에 추가
                                    if os.path.exists(file_path):
                                        files_to_attach.append(discord.File(file_path))
                                    else:
                                        print(f"File not found: {file_path}")
                                        
                                print(files_to_attach)
                                if files_to_attach:
                                    await discord_channel.send(files=files_to_attach)
                                else:
                                    await discord_channel.send(f"No valid image files found. - {author_info}")

                        elif message_type == "text":
                            # 일반 텍스트 메시지 처리
                            text_answers = message_data.get("text_answers")
                            if text_answers:
                                await discord_channel.send(text_answers)
                        else:
                            print(f"Unknown message type: {message_type}")
                    else:
                        print(f"Channel ID {channel_id} not found")


# 디스코드 클라이언트 설정
# intents는 Discord API에서 특정 이벤트 유형에 대한 접근을 허용하거나 제한하는 설정을 관리하는 개념. 
# 봇이 어떤 정보나 이벤트에 접근할 수 있는지를 제어하는 역할
intents = discord.Intents.default()
intents.message_content = True  # 메시지 내용을 읽을 수 있도록 설정

client = discord.Client(intents=intents)

# on_ready() 함수를 on_ready 이벤트에 연결시킵니다. 즉, 봇이 Discord 서버에 성공적으로 연결되었을 때 on_ready 이벤트가 발생하고, 이 함수가 실행
@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    # '!질문'으로 시작하는 메시지 파싱
    if message.content.startswith('!백') or message.content.startswith('!운동'):
        # !질문 제거 후 이후 내용만 뽑음
        content = message.content
        if content:
            # 예시: '!질문 아시아의 면적은? 우선순위:5'
            if "/우선" in content:
                question = content.replace('/우선', '')
                priority = 5  # 우선순위 설정
            # 기본 우선순위는 낮게 설정
            else:
                question = content
                priority = 2  # 기본 우선순위
            
            # RabbitMQ로 질문과 작성자 이름, 채널 ID, 우선순위 전송 (in_queue로 전송)
            await send_to_rabbitmq(question, priority, message.author.name, message.channel.id)
            await message.reply(f'질문이 접수되었습니다')

            # 비동기적으로 out_queue에서 답변 받기 (한 번만 실행)
            asyncio.ensure_future(consume_out_queue())
        else:
            await message.reply('질문 내용을 써주세요')
            
    elif message.content.startswith('!인물'):
        content = message.content
        content=content[3:].strip()
        # print(content)
        if content:
            if content =='임하늘':
                await message.reply(f'할재 개그의 1인자, 책을 나눠주는 따뜻한 마음을 가지고 있음')
            elif content =='박광현':
                await message.reply(f'설데의 보물, 미래의 잡스, 관악산 신령으로 불림. 관악산에 서식하며 쓰레기 버리는 사람을 스틱으로 후드려팸')
            elif content =='윤성준':
                await message.reply(f'설데의 리더. 크래프톤의 많은 우리사주를 산 뒤 폭락. 그 이후 흑화해서 회사의 맥주를 신나게 털어먹으며 술꾼이됨')
            elif content =='이상혁':
                await message.reply(f'생선을 수렵하는 초기 문명의 인류. 이진욱과 닮은 외모를 가지고 있고, 치명적인 전갈독을 가지고 있음')
            elif content =='이정훈':
                await message.reply(f'설데의 스윗가이. 최면의 1인자. 결혼 후 행복하다고 강력한 자기 세뇌에 성공했음')
            elif content =='장태수':
                await message.reply(f'보기보다 차갑지 않고 따뜻한 남자. 사진에 재능이 있어 여행에 데려가면 인생사진 3장 이상은 건져준다. 좋아하는 야구 팀은 두산, 그러나 잠실의 주인은 LG이다.')
            elif content =='이서림':
                await message.reply(f'2기의 네임드 능력자. 맥북을 너무 자랑하는 자랑쟁이. 좋아하는 야구팀은 LG, 그러나 잠실의 주인은 두산이다.')
            # elif content =='이서림':
            #     await message.reply(f'2기의 네임드 능력자. 맥북을 너무 자랑하는 자랑쟁이. 좋아하는 야구팀은 LG, 그러나 잠실의 주인은 두산이다.')
            else:
                await message.reply(f'정보를 업데이트 중입니다.')
        # 작성자가 'igeolwaehani'이면 모든 이모지를 추가
        
    if message.author.name == 'igeolwaehani':
        # 기본 이모지 리스트 (원하는 대로 수정 가능)
        all_emojis = ['👍', '😂', '🎉', '😎', '🔥', '❤️', '🙌', '👏', '😍', '🎈']

        # 모든 이모지를 메시지에 리액션으로 추가
        for emoji in all_emojis:
            try:
                await message.add_reaction(emoji)
            except discord.HTTPException:
                print(f"이모지 {emoji} 추가 실패")

        # '넘 멋있으세요!'라는 댓글 남기기
        # await message.reply('넘 멋있으세요!')

# 디스코드 봇 실행 (토큰을 디스코드 개발자 포털에서 발급받아야 함)
client.run(dis_token)

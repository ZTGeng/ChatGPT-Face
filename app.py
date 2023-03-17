import os
from apscheduler.schedulers.background import BackgroundScheduler
import time
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
import openai
from google.cloud import texttospeech
import uuid
from markupsafe import escape
from wav2lip.wav2lip import FaceVideoMaker
import pytz

work_dir = 'temp'
if not os.path.exists(work_dir):
    os.mkdir(work_dir)
files_to_delete = []

print('服务器初始化...')
app = Flask(__name__)
faceVideoMaker = FaceVideoMaker(audio_dir=work_dir, video_dir=work_dir)
start_time = time.time()

# OPENAI 相关
# OPENAI token 使用统计
openai.api_key = os.getenv("OPENAI_API_KEY")

site_api_key_completion_tokens = 0
site_api_key_prompt_tokens = 0
site_api_key_total_tokens = 0
site_api_key_requests = 0
custom_api_key_completion_tokens = 0
custom_api_key_prompt_tokens = 0
custom_api_key_total_tokens = 0
custom_api_key_requests = 0

def print_log():
    running_time = time.time() - start_time
    print('运行持续时间：' + str(int(running_time / 3600)) + ' 小时 ' + str(int(running_time % 3600 / 60)) + ' 分钟 ' + str(int(running_time % 60)) + ' 秒')
    print('------------------------------------')
    print('本站 api-key          总对话数: ' + str(site_api_key_requests))
    print('本站 api-key completion tokens: ' + str(site_api_key_completion_tokens))
    print('本站 api-key     prompt tokens: ' + str(site_api_key_prompt_tokens))
    print('本站 api-key      total tokens: ' + str(site_api_key_total_tokens))
    print('------------------------------------')
    print('外部 api-key          总对话数: ' + str(custom_api_key_requests))
    print('外部 api-key completion tokens: ' + str(custom_api_key_completion_tokens))
    print('外部 api-key     prompt tokens: ' + str(custom_api_key_prompt_tokens))
    print('外部 api-key      total tokens: ' + str(custom_api_key_total_tokens))

# OPENAI API-KEY 访问限制
API_LIMIT_PER_HOUR = 1000
counter = 0

def reset_counter():
    global counter
    counter = 0

def add_counter():
    global counter
    counter += 1

def is_limit_reached():
    global counter
    return counter >= API_LIMIT_PER_HOUR

# 定时任务
def hourly_maintain():
    reset_counter()
    print_log()

    for file in files_to_delete:
        os.remove(file)
    files_to_delete.clear()
    for filename in os.listdir(work_dir):
        file = os.path.join(work_dir, filename)
        if os.path.isfile(file):
            files_to_delete.append(file)

scheduler = BackgroundScheduler(timezone=pytz.utc)
next_hour_time = datetime.fromtimestamp(time.time() + 3600 - time.time() % 3600)
scheduler.add_job(hourly_maintain, 'interval', minutes=60, next_run_time=next_hour_time)
scheduler.start()

# OPENAI API 调用
def fetch_chat_response(text, api_key):
    return openai.ChatCompletion.create(
        api_key=api_key,
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": text},
        ],
        # max_tokens=2000
    )

def parse_chat_response(response, useSiteApiKey):
    # print(response)
    message = response['choices'][0]['message']['content']
    # print(message)
    if useSiteApiKey:
        global site_api_key_completion_tokens
        global site_api_key_prompt_tokens
        global site_api_key_total_tokens
        global site_api_key_requests
        site_api_key_completion_tokens += response['usage']['completion_tokens']
        site_api_key_prompt_tokens += response['usage']['prompt_tokens']
        site_api_key_total_tokens += response['usage']['total_tokens']
        site_api_key_requests += 1
    else:
        global custom_api_key_completion_tokens
        global custom_api_key_prompt_tokens
        global custom_api_key_total_tokens
        global custom_api_key_requests
        custom_api_key_completion_tokens += response['usage']['completion_tokens']
        custom_api_key_prompt_tokens += response['usage']['prompt_tokens']
        custom_api_key_total_tokens += response['usage']['total_tokens']
        custom_api_key_requests += 1
    return message

def remove_code_block(message):
    if message.find('```') == -1:
        return message
    else:
        return ''.join([words for i, words in enumerate(message.split('```')) if i % 2 == 0])

# GOOGLE Text-to-Speech 相关
textToSpeechClient = texttospeech.TextToSpeechClient()
voice = texttospeech.VoiceSelectionParams(
    language_code="zh-CN",
    ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
)
audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
    speaking_rate=1.25
)

def text_to_speech(text, filename):
    # test_time = time.time()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    response = textToSpeechClient.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config)
    audio_path = os.path.join(work_dir, f'{filename}.wav')
    with open(audio_path, "wb") as out:
        out.write(response.audio_content)
        # print('TTS 测试时间：' + str(time.time() - test_time))

# Flask 路由
@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/message', methods=['POST'])
def message():
    data = request.json
    # print(data)
    useSiteApiKey = data.get('key_type') == 'default'
    if useSiteApiKey:
        if is_limit_reached():
            # Error code 1: 本站 api-key 超过使用限制
            return jsonify({'error_code': 1}), 200
        
    try:
        if useSiteApiKey:
            response = fetch_chat_response(data.get('message'), openai.api_key)
            add_counter()
        else:
            api_key = data.get('api_key')
            if not api_key:
                # Error code 2: 外部 api-key 为空或无效
                return jsonify({'error_code': 2}), 200
            response = fetch_chat_response(data.get('message'), api_key)
    except openai.error.AuthenticationError as e:
        # Error code 1: 本站 api-key 欠费或无效，按照超过使用限制处理
        # （因为本站 api-key 的任何信息都应避免暴露，所以相关错误统一返回超过使用限制）
        # Error code 2: 外部 api-key 为空或无效
        return jsonify({'error_code': 1 if useSiteApiKey else 2}), 200
    except openai.error.RateLimitError as e:
        # Error code 3: api-key 一定时间内使用太过频繁
        return jsonify({'error_code': 3}), 200
    except (openai.error.APIError, openai.error.Timeout) as e:
        # Error code 4: OPENAI API 服务异常
        return jsonify({'error_code': 4}), 200
    except Exception as e:
        # Error code 0: 未知错误
        return jsonify({'error_code': 0, 'message': e}), 200
    
    message = parse_chat_response(response, useSiteApiKey)

    message_no_code_block = remove_code_block(message)
    id = str(uuid.uuid4())[:8]
    text_to_speech(message_no_code_block, id)
    faceVideoMaker.makeVideo(id)

    return jsonify({'message': escape(message), 'video_url': f'/{work_dir}/{id}.mp4'}), 200

@app.route(f'/{work_dir}/<path:filename>')
def get_video_files(filename):
    return send_from_directory(os.path.join(os.getcwd(), work_dir), filename)

@app.route('/api/face_img')
def face_img():
    return send_from_directory(os.path.join(os.getcwd(), 'assets'), 'face_2.jpg')


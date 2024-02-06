import os, time, uuid
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
import openai, tiktoken
from openai import OpenAI
from google.cloud import texttospeech
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
client = OpenAI()
model_35_name = 'gpt-3.5-turbo'
model_4_name = 'gpt-4-turbo-preview'
model_35_token_limit = 16385
model_4_token_limit = 128000

# OPENAI token 使用统计
# openai.api_key = os.getenv("OPENAI_API_KEY")

site_api_key_completion_tokens_model_35 = 0
site_api_key_prompt_tokens_model_35 = 0
site_api_key_total_tokens_model_35 = 0
site_api_key_requests_model_35 = 0
site_api_key_completion_tokens_model_4 = 0
site_api_key_prompt_tokens_model_4 = 0
site_api_key_total_tokens_model_4 = 0
site_api_key_requests_model_4 = 0
custom_api_key_completion_tokens = 0
custom_api_key_prompt_tokens = 0
custom_api_key_total_tokens = 0
custom_api_key_requests = 0

def print_log():
    running_time = time.time() - start_time
    print('运行持续时间：' + str(int(running_time / 3600)) + ' 小时 ' + str(int(running_time % 3600 / 60)) + ' 分钟 ' + str(int(running_time % 60)) + ' 秒')
    print('------------------------------------')
    print('本站 api-key GPT3.5     conversations: ' + str(site_api_key_requests_model_35))
    print('本站 api-key GPT3.5 completion tokens: ' + str(site_api_key_completion_tokens_model_35))
    print('本站 api-key GPT3.5     prompt tokens: ' + str(site_api_key_prompt_tokens_model_35))
    print('本站 api-key GPT3.5      total tokens: ' + str(site_api_key_total_tokens_model_35))
    print('------------------------------------')
    print('本站 api-key GPT4       conversations: ' + str(site_api_key_requests_model_4))
    print('本站 api-key GPT4   completion tokens: ' + str(site_api_key_completion_tokens_model_4))
    print('本站 api-key GPT4       prompt tokens: ' + str(site_api_key_prompt_tokens_model_4))
    print('本站 api-key GPT4        total tokens: ' + str(site_api_key_total_tokens_model_4))
    print('------------------------------------')
    print('外部 api-key           conversations: ' + str(custom_api_key_requests))
    print('外部 api-key       completion tokens: ' + str(custom_api_key_completion_tokens))
    print('外部 api-key           prompt tokens: ' + str(custom_api_key_prompt_tokens))
    print('外部 api-key            total tokens: ' + str(custom_api_key_total_tokens))

# OPENAI API-KEY 访问限制
TOKEN_LIMIT_DAILY_MODEL_35 = 1000000
TOKEN_LIMIT_DAILY_MODEL_4 = 128000
token_usage_daily_model_35 = 0
token_usage_daily_model_4 = 0

def reset_counter():
    global token_usage_daily_model_35
    global token_usage_daily_model_4
    token_usage_daily_model_35 = 0
    token_usage_daily_model_4 = 0

def update_counter(is_model_4, tokens_used):
    if is_model_4:
        global token_usage_daily_model_4
        token_usage_daily_model_4 += tokens_used
    else:
        global token_usage_daily_model_35
        token_usage_daily_model_35 += tokens_used

def is_limit_reached(is_model_4):
    if is_model_4:
        return token_usage_daily_model_4 >= TOKEN_LIMIT_DAILY_MODEL_4
    else:
        return token_usage_daily_model_35 >= TOKEN_LIMIT_DAILY_MODEL_35

# 定时任务
def daily_maintain():
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
next_day_time = datetime.fromtimestamp(time.time() + 86400 - time.time() % 86400)
scheduler.add_job(daily_maintain, 'interval', minutes=3600, next_run_time=next_day_time)
scheduler.start()

# Token counting 相关
# 详见 https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
encoding_model_35 = tiktoken.encoding_for_model(model_35_name)
encoding_model_4 = tiktoken.encoding_for_model(model_4_name)

def encode(text: str, use_model_4):
    return encoding_model_4.encode(text) if use_model_4 else encoding_model_35.encode(text)

def num_tokens_from_string(text: str, use_model_4):
    return len(encode(text, use_model_4))

def num_tokens_from_messages(messages, use_model_4):
    tokens_per_message = 3
    tokens_per_name = 1
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += num_tokens_from_string(value, use_model_4)
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def tokens_remaining(messages, use_model_4):
    if use_model_4:
        return model_4_token_limit - num_tokens_from_messages(messages, use_model_4)
    else:
        return model_35_token_limit - num_tokens_from_messages(messages, use_model_4)

# OPENAI API 调用
def fetch_chat_response(use_model_4, messages, temperature=1, max_tokens=None, presence_penalty=0, frequency_penalty=0, stop=None, logit_bias=None, external_api_key=None):
    chat_completion_args = {
        'model': model_4_name if use_model_4 else model_35_name,
        'messages': messages,
    }
    if temperature != 1:
        chat_completion_args['temperature'] = temperature
    if max_tokens is not None:
        max_tokens = min(int(max_tokens), tokens_remaining(messages, use_model_4))
        chat_completion_args['max_tokens'] = max_tokens
    if presence_penalty != 0:
        chat_completion_args['presence_penalty'] = presence_penalty
    if frequency_penalty != 0:
        chat_completion_args['frequency_penalty'] = frequency_penalty
    if stop is not None:
        chat_completion_args['stop'] = stop
    if logit_bias is not None:
        logit_bias = { id: value for key, value in logit_bias.items() for id in encode(key, use_model_4) }
        chat_completion_args['logit_bias'] = logit_bias

    if external_api_key is None:
        return client.chat.completions.create(**chat_completion_args)
    else:
        temp_client = OpenAI(api_key=external_api_key)
        return temp_client.chat.completions.create(**chat_completion_args)

def parse_chat_response(response, use_model_4, use_site_api_key):
    message = response.choices[0].message.content
    if response.usage is not None:
        if use_site_api_key:
            update_counter(use_model_4, (response.usage.total_tokens or 0))
            if use_model_4:
                global site_api_key_completion_tokens_model_4
                global site_api_key_prompt_tokens_model_4
                global site_api_key_total_tokens_model_4
                global site_api_key_requests_model_4
                site_api_key_completion_tokens_model_4 += (response.usage.completion_tokens or 0)
                site_api_key_prompt_tokens_model_4 += (response.usage.prompt_tokens or 0)
                site_api_key_total_tokens_model_4 += (response.usage.total_tokens or 0)
                site_api_key_requests_model_4 += 1
            else:
                global site_api_key_completion_tokens_model_35
                global site_api_key_prompt_tokens_model_35
                global site_api_key_total_tokens_model_35
                global site_api_key_requests_model_35
                site_api_key_completion_tokens_model_35 += (response.usage.completion_tokens or 0)
                site_api_key_prompt_tokens_model_35 += (response.usage.prompt_tokens or 0)
                site_api_key_total_tokens_model_35 += (response.usage.total_tokens or 0)
                site_api_key_requests_model_35 += 1
        else:
            global custom_api_key_completion_tokens
            global custom_api_key_prompt_tokens
            global custom_api_key_total_tokens
            global custom_api_key_requests
            custom_api_key_completion_tokens += (response.usage.completion_tokens or 0)
            custom_api_key_prompt_tokens += (response.usage.prompt_tokens or 0)
            custom_api_key_total_tokens += (response.usage.total_tokens or 0)
            custom_api_key_requests += 1
    return message

def remove_code_block(message):
    if message.find('```') == -1:
        return message
    else:
        return ''.join([words for i, words in enumerate(message.split('```')) if i % 2 == 0])

# GOOGLE Text-to-Speech 相关
textToSpeechClient = texttospeech.TextToSpeechClient()
voice_en = texttospeech.VoiceSelectionParams(
    language_code="en-US",
    name="en-US-Wavenet-F",
    ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
)
voice_zh = texttospeech.VoiceSelectionParams(
    language_code="cmn-CN",
    name="cmn-CN-Wavenet-D",
    ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
)
audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
    speaking_rate=1.25
)

def text_to_speech(text, lang, filename):
    # test_time = time.time()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = voice_zh if lang == 'zh' else voice_en
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
def message_v3():
    data = request.json
    use_model_4 = data.get('use_model_4', False)
    use_site_api_key = data.get('key_type', '').lower() == 'default'
    if use_site_api_key:
        if is_limit_reached(use_model_4):
            # Error code 1: 本站 api-key 超过使用限制
            return jsonify({'error_code': 1}), 200
    messages = data.get('messages')
    if tokens_remaining(messages, use_model_4) <= 0:
        # Error code 5: 消息过长
        return jsonify({'error_code': 5}), 200
    request_params = {
        'messages': messages,
    }
    temperature = data.get('temperature')
    if temperature:
        request_params['temperature'] = int(temperature)
    max_tokens = data.get('max_tokens')
    if max_tokens:
        request_params['max_tokens'] = int(max_tokens)
    presence_penalty = data.get('presence_penalty')
    if presence_penalty:
        request_params['presence_penalty'] = float(presence_penalty)
    frequency_penalty = data.get('frequency_penalty')
    if frequency_penalty:
        request_params['frequency_penalty'] = float(frequency_penalty)
    stop_sequences = data.get('stop_sequences')
    if stop_sequences:
        request_params['stop'] = stop_sequences
    logit_bias = data.get('logit_bias')
    if logit_bias:
        request_params['logit_bias'] = logit_bias
    try:
        if use_site_api_key:
            response = fetch_chat_response(use_model_4, **request_params)
        else:
            api_key = data.get('api_key')
            if not api_key:
                # Error code 2: 外部 api-key 为空或无效
                return jsonify({'error_code': 2}), 200
            request_params['external_api_key'] = api_key
            response = fetch_chat_response(use_model_4, **request_params)
    except openai.AuthenticationError as e:
        # Error code 1: 本站 api-key 欠费或无效，按照超过使用限制处理
        # （因为本站 api-key 的任何信息都应避免暴露，所以相关错误统一返回超过使用限制）
        # Error code 2: 外部 api-key 为空或无效
        return jsonify({'error_code': 1 if use_site_api_key else 2}), 200
    except openai.RateLimitError as e:
        # Error code 3: api-key 一定时间内使用太过频繁
        return jsonify({'error_code': 3}), 200
    except (openai.APIError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError) as e:
        # Error code 4: OPENAI API 服务异常
        return jsonify({'error_code': 4}), 200
    except Exception as e:
        # Error code 0: 未知错误
        # e.g. openai.BadRequestError
        print(e)
        return jsonify({'error_code': 0, 'message': str(e)}), 200
    
    message = parse_chat_response(response, use_model_4, use_site_api_key)

    is_video_mode = data.get('video')
    if not is_video_mode:
        return jsonify({'message': escape(message)}), 200

    lang = data.get('language_code')
    message_no_code_block = remove_code_block(message)
    id = str(uuid.uuid4())[:8]
    text_to_speech(message_no_code_block, lang, id)
    faceVideoMaker.makeVideo(id)

    return jsonify({'message': escape(message), 'video_url': f'/{work_dir}/{id}.mp4'}), 200

@app.route('/api/tokens', methods=['POST'])
def get_tokens():
    data = request.json
    use_model_4 = data.get('use_model_4', False)
    messages = data.get('messages')
    return jsonify({'tokens': num_tokens_from_messages(messages, use_model_4)}), 200

@app.route(f'/{work_dir}/<path:filename>')
def get_video_files(filename):
    return send_from_directory(os.path.join(os.getcwd(), work_dir), filename)

@app.route('/api/face_img')
def face_img():
    return send_from_directory(os.path.join(os.getcwd(), 'assets'), 'face_2.jpg')

@app.route('/<path:filename>')
def static_html(filename):
    return app.send_static_file(filename)


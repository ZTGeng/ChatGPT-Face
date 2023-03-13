import os, subprocess
from apscheduler.schedulers.background import BackgroundScheduler
import time
from datetime import datetime
from flask import Flask, request, jsonify, Response, send_from_directory
import openai
from google.cloud import texttospeech
import uuid
from markupsafe import escape

audio_dir = 'audio'

app = Flask(__name__)
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
    print_log()

def add_counter():
    global counter
    counter += 1

def is_limit_reached():
    global counter
    return counter >= API_LIMIT_PER_HOUR

scheduler = BackgroundScheduler()
next_hour_time = datetime.fromtimestamp(time.time() + 3600 - time.time() % 3600)
scheduler.add_job(reset_counter, 'interval', minutes=60, next_run_time=next_hour_time)
scheduler.start()

# OPENAI API 调用
def fetch_chat_response(text, api_key):
    return openai.ChatCompletion.create(
        api_key=api_key,
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": text},
        ],
        max_tokens=2000
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
    with open("temp\\" + filename + ".wav", "wb") as out:
        out.write(response.audio_content)
        # print('TTS 测试时间：' + str(time.time() - test_time))

# Flask 路由

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/message', methods=['POST'])
def message():
    data = request.json
    print(data)
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
    # text_to_speech(message_no_code_block, id)

    return jsonify({'message': escape(message)}), 200

@app.route('/temp/<path:filename>')
def get_temp_files(filename):
    return send_from_directory(os.path.join(os.getcwd(), 'temp'), filename)

@app.route('/api/video')
def video():
    def generate():
        # 打开MP4文件
        with open('temp/output200_fps15.mp4', 'rb') as f:
            # 调用FFmpeg命令行工具生成HLS流
            # 这里的命令参数是示例，具体的参数需要根据实际情况调整
            cmd = ['ffmpeg', '-i', '-', '-c:v', 'libx264', '-c:a', 'aac', '-hls_list_size', '10', '-hls_time', '10', '-f', 'hls', 'pipe:1']
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # 读取MP4文件并写入到FFmpeg进程的输入流
            while True:
                data = f.read(1024)
                print("read")
                if not data:
                    break
                process.stdin.write(data)
            process.stdin.close()
            # 从FFmpeg进程的输出流读取HLS流并传输到客户端
            while True:
                data = process.stdout.read(1024)
                print("write")
                if not data:
                    break
                yield data

    # 设置HTTP响应头，指定MIME类型为application/vnd.apple.mpegurl
    headers = {
        'Content-Type': 'application/vnd.apple.mpegurl',
        'Content-Disposition': 'attachment; filename="video.m3u8"'
    }
    # 返回一个生成器作为响应体
    return Response(generate(), headers=headers)

@app.route('/facetest')
def test():
    p1 = open('static/face_1.jpg', 'rb').read()
    p2 = open('static/face_2.jpg', 'rb').read()
    def generator():
        
        for i in range(20):
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + (p1 if i % 2 == 0 else p2) + b'\r\n')
            time.sleep(0.5)
        yield b'--frame--\r\n'
    return Response(generator(), mimetype='multipart/x-mixed-replace; boundary=frame')
        

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
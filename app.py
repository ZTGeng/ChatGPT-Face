import os
from apscheduler.schedulers.background import BackgroundScheduler
import time
from datetime import datetime
from flask import Flask, request, jsonify
import openai

app = Flask(__name__)
start_time = time.time()

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

def chat_completion(text, api_key):
    return openai.ChatCompletion.create(
        api_key=api_key,
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": text},
        ],
        max_tokens=2000
    )

def parseResponse(response, useSiteApiKey):
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
    return jsonify({'message': message}), 200

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
            return jsonify({'error_code': 1}), 200
        
    try:
        if useSiteApiKey:
            response = chat_completion(data.get('message'), openai.api_key)
            add_counter()
        else:
            api_key = data.get('api_key')
            if not api_key:
                return jsonify({'error_code': 2}), 200
            response = chat_completion(data.get('message'), api_key)
    except openai.error.AuthenticationError as e:
        return jsonify({'error_code': 1 if useSiteApiKey else 2}), 200
    except openai.error.RateLimitError as e:
        return jsonify({'error_code': 3}), 200
    except (openai.error.APIError, openai.error.Timeout) as e:
        return jsonify({'error_code': 4}), 200
    except Exception as e:
        return jsonify({'error_code': 0, 'message': e}), 200
    return parseResponse(response, useSiteApiKey)
        
# @app.route('/ws')
# def websocket():
#     data = request.json
#     print(data)
#     return jsonify({'message': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
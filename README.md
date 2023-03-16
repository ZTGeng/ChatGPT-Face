# ChatGPT - Text-to-Speech - Wav2lip

[English](./README_en.md)

给ChatGPT的聊天模式加上语音和自动生成的视频。

## 安装依赖项

Python3.9。(3.8以下版本未测试。3.10以上不支持。)

使用virtualenv或者venv创建并激活虚拟环境。

运行：

    pip install -r requirements.txt

## 配置环境文件 .env

复制项目根目录下的文件 '.envsample' 为 '.env'（注意文件名前面的 '.'）。

将从 openai.com 获得的 API-KEY 复制粘贴给 OPENAI_API_KEY 变量。

按照 [Google Cloud Text-to-Speech](https://cloud.google.com/text-to-speech/docs/before-you-begin?hl=zh-cn) 的指引，为你的谷歌开发者账号启用 Text-to-Speech API、创建服务账号，最后创建 JSON 密钥，并将密钥保存在 `keys/` 目录下（或本项目文件夹下任何安全的目录下，不要忘记将保存路径添加到 .gitignore 中）。然后将密钥路径和文件名复制粘贴给 '.env' 文件中 GOOGLE_APPLICATION_CREDENTIALS 变量。

例如，你的密钥文件保存在：项目文件夹/keys/google-cloud-text-to-speech-key.json，那么应该设置`GOOGLE_APPLICATION_CREDENTIALS="keys/google-cloud-text-to-speech-key.json"`。

注意！不要暴露你的 openai API-KEY 和 Google Cloud JSON 密钥。

## 下载 Wav2lip 预训练模型

前往 [Wav2Lip 的 Github 页面](https://github.com/Rudrabha/Wav2Lip)，在 **Getting the weights** 部分，下载 Model 名为 “Wav2Lip + GAN”的预训练模型，其链接为 [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW)，文件名为 `wav2lip_gan.pth`。将其保存在项目文件夹下的 `wav2lip/weights/`目录下。

## 启动 Flask 服务器

    flask run

（不要使用 `python app.py`，那将无法加载 .env 环境文件。）

## Acknowledgements

```
@inproceedings{10.1145/3394171.3413532,
author = {Prajwal, K R and Mukhopadhyay, Rudrabha and Namboodiri, Vinay P. and Jawahar, C.V.},
title = {A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild},
year = {2020},
isbn = {9781450379885},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3394171.3413532},
doi = {10.1145/3394171.3413532},
booktitle = {Proceedings of the 28th ACM International Conference on Multimedia},
pages = {484–492},
numpages = {9},
keywords = {lip sync, talking face generation, video generation},
location = {Seattle, WA, USA},
series = {MM '20}
}
```

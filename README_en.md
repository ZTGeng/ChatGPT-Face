# ChatGPT - Text-to-Speech - Wav2lip

[中文](./README.md)

Video chatting with GPT by Text-to-Speech transferring and video generating.

## Requirements

Python3.9. (Pre 3.8: not tested. Post 3.10: not usable.)

Use virtualenv or venv. Activate

### Check if PyTorch needs to be installed separately

Go to [PyTorch Get Started](https://pytorch.org/get-started/locally/), select your os and compute platform (CUDA vs CPU), check the command line below. If you see `pip3 install torch torchvision torchaudio`, you don't need to run it, because requirements.txt will take care. If you see a command line with arguments, you need to run it first (you can remove `torchvision torchaudio` from the command line).

### Run:

    pip install -r requirements.txt

### Install ffmpeg

Linux:

    sudo apt install ffmpeg

Windows, Mac:

Download and install, and add the path to the environment varibles PATH.

## Setup the .env file

Copy the '.envsample' file and rename it as '.env' (don't miss the leading dot).

Copy the API-KEY you get from openai.com and paste after OPENAI_API_KEY variable.

Follow the guide at [Google Cloud Text-to-Speech](https://cloud.google.com/text-to-speech/docs/before-you-begin?hl=zh-cn) and setup your Google developer account by turning on Text-to-Speech API, creating a servive account, and finally creating a private JSON key. Save the JSON key in the `keys/` directory (or any other directories in this repo to keep it safe, just don't forget to add the directory to the .gitignore). Then paste the path and the key filename to GOOGLE_APPLICATION_CREDENTIALS variable in the '.env' file.

For example, the JSON key is saved at: THIS_REPO/keys/google-cloud-text-to-speech-key.json, you should set `GOOGLE_APPLICATION_CREDENTIALS="keys/google-cloud-text-to-speech-key.json"`.

Important! DO NOT expose you openai API-KEY and Google Cloud JSON key.

## Download Wav2lip pre-trained model

Go to [Wav2Lip Github Page](https://github.com/Rudrabha/Wav2Lip), find **Getting the weights** section, download the pre-trained Model named “Wav2Lip + GAN, whose link is [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW), and filename is `wav2lip_gan.pth`. Save it under `wav2lip/weights/` directory.

## Run Flask server

    flask run

(Don't use `python app.py` or you will not load the .env file.)

Open in your browser:

    127.0.0.1:5000

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

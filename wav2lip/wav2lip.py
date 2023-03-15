import os, subprocess, platform
import numpy as np
import cv2
import torch
from . import audio
from .models import Wav2Lip

class FaceVideoMaker(object):
    def __init__(self, weights_file='wav2lip/weights/wav2lip_gan.pth', face_img='assets/face_200.png', coords=(10, 200, 49, 169), y1a=110, audio_dir='temp', video_dir='temp', fps=15, device='cpu'):
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.device = device
        self.fps = fps
        self.frame = cv2.imread(face_img)
        self.y1, self.y2, self.x1, self.x2 = coords
        self.y1a = y1a
        self.img_size = 96
        self.mel_step_size = 16
        self.wav2lip_batch_size = 128
        self.face = self.frame[self.y1:self.y2, self.x1:self.x2]
        self.face = cv2.resize(self.face, (self.img_size, self.img_size))

        weights_path = os.path.join(os.getcwd(), weights_file)
        weights = torch.load(weights_path, map_location=torch.device(self.device))
        s = weights["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model = Wav2Lip()
        model.load_state_dict(new_s)
        model = model.to(device)
        self.model = model.eval()

    def makeVideo(self, id):
        audio_path = os.path.join(os.getcwd(), self.audio_dir, f'{id}.wav')
        wav = audio.load_wav(audio_path, 16000)
        mel = audio.melspectrogram(wav)
        mel_chunks = []
        mel_idx_multiplier = 80./self.fps 
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + self.mel_step_size])
            i += 1
        
        frame_h, frame_w = self.frame.shape[:-1]
        video_path = os.path.join(os.getcwd(), self.video_dir, f'{id}.avi')
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), self.fps, (frame_w, frame_h))
        for (img_batch, mel_batch) in self.datagen2(self.face, mel_chunks):
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)

            with torch.no_grad():
                pred = self.model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            for p in pred:
                f = self.frame.copy()
                p = cv2.resize(p.astype(np.uint8), (self.x2 - self.x1, self.y2 - self.y1))
                f[self.y1a:self.y2, self.x1:self.x2] = p[self.y1a-self.y1:]
                out.write(f)
        out.release()

        face_video_path = os.path.join(os.getcwd(), self.video_dir, f'{id}.mp4')
        command = f'ffmpeg -y -i {audio_path} -i {video_path} -strict -2 -q:v 1 {face_video_path}'
        subprocess.call(command, shell=platform.system() != 'Windows')
        os.remove(audio_path)
        os.remove(video_path)

    def datagen2(self, face, mels):
        img_batch, mel_batch = [], []

        for m in mels:
            img_batch.append(face.copy())
            mel_batch.append(m)

            if len(img_batch) >= self.wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, self.img_size//2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch
                img_batch, mel_batch = [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, self.img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch

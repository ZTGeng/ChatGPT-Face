import os, subprocess, platform
import numpy as np
import cv2
import torch
from . import audio
from .models import Wav2Lip

class FaceVideoMaker(object):
    # 图片坐标使用 magic numbers 以简化计算。格式为 (y1, y2, x1, x2)。其中：
    # - 图片左上角为 (0, 0)。人脸占据图片中一个长方形区域，其左上角坐标为 (x1, y1)，右下角坐标为 (x2, y2)。
    # y1r 是实际替换时使用的 y1 坐标，默认约为 y1 与 y2 的中点。因为静态人脸图片的上半区在说话时几乎不发生改变，所以可只替换下半区。如果 y1r 为 None，则使用 y1。
    # 如果需要替换图片，可以（但没必要）使用一个叫 face_detection 的模型来检测新图片中人脸的位置以计算坐标。
    # face_detection 的代码详见：https://github.com/1adrianb/face-alignment，或见原 Wave2Lip 库中的引用：https://github.com/Rudrabha/Wav2Lip
    def __init__(self, weights_file='wav2lip/weights/wav2lip_gan.pth', face_img='assets/face_200.png', coords=(34, 161, 51, 147), y1r=91, audio_dir='temp', video_dir='temp', fps=15):
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.device = 'cuda' if (torch.cuda.is_available()) else 'cpu'
        self.fps = fps
        self.frame = cv2.imread(face_img)
        self.y1, self.y2, self.x1, self.x2 = coords
        self.y1r = y1r if y1r else self.y1
        self.img_size = 96
        self.mel_step_size = 16
        self.wav2lip_batch_size = 128
        self.face = self.frame[self.y1:self.y2, self.x1:self.x2]
        self.face = cv2.resize(self.face, (self.img_size, self.img_size))

        weights_path = os.path.join(os.getcwd(), weights_file)
        print('加载模型于', self.device, '...')
        weights = torch.load(weights_path, map_location=torch.device(self.device))
        s = weights["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model = Wav2Lip()
        model.load_state_dict(new_s)
        model = model.to(self.device)
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
                f[self.y1r:self.y2, self.x1:self.x2] = p[self.y1r-self.y1:]
                out.write(f)
        out.release()

        face_video_path = os.path.join(os.getcwd(), self.video_dir, f'{id}.mp4')
        command = f'ffmpeg -y -i {audio_path} -i {video_path} -strict -2 -q:v 1 {face_video_path} -loglevel error'
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

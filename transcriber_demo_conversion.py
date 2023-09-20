import io
import os
import speech_recognition as sr
import whisper
import torch
import numpy as np
from scipy.io import wavfile
import asyncio

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform

class Transcriber:
    def __init__(self, model="base", record_timeout=2, phrase_timeout=3, energy_threshold=1000, default_microphone='pulse') -> None:
        self.model = model
        self.energy_threshold = energy_threshold
        self.record_timeout = record_timeout
        self.phrase_timeout = phrase_timeout
        self.default_microphone = default_microphone

        self.phrase_time = None
        self.last_sample = bytes()
        self.data_queue = Queue()
        self.source = None

        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = energy_threshold
        self.recorder.dynamic_energy_threshold = False
        self.transcription = ['']

    def load_model(self):
        if self.model != "large":
            self.model = self.model + ".en"
        self.audio_model = whisper.load_model(self.model)

    def execute(self):
        asyncio.run(self.__execute__())

    async def __execute__(self):
        print("Model loaded.\n")
        while True:
            try:
                now = datetime.utcnow()
                if not self.data_queue.empty():
                    phrase_complete = False
                    if self.phrase_time and now - self.phrase_time > timedelta(seconds=self.phrase_timeout):
                        self.last_sample = bytes()
                        phrase_complete = True
                    self.phrase_time = now

                    while not self.data_queue.empty():
                        data = self.data_queue.get()
                        self.last_sample += data

                    audio_data = sr.AudioData(self.last_sample, self.source.SAMPLE_RATE, self.source.SAMPLE_WIDTH)
                    wav_data = io.BytesIO(audio_data.get_wav_data())
                    wav_data.seek(0)

                    _, in_mem_wav_data = wavfile.read(wav_data)
                    in_mem_wav_data = (in_mem_wav_data.astype(float) / np.iinfo(np.int16).max).astype(np.float32)

                    # Read the transcription.
                    result = self.audio_model.transcribe(in_mem_wav_data, fp16=torch.cuda.is_available())
                    text = result['text'].strip()

                    if phrase_complete:
                        self.transcription.append(text)
                    else:
                        self.transcription[-1] = text

                    os.system('cls' if os.name=='nt' else 'clear')
                    for line in self.transcription:
                        print(line)
                    print('', end='', flush=True)

                    #sleep(0.1)
                    await asyncio.sleep(0.1)
            except KeyboardInterrupt:
                break

        print("\n\nTranscription:")
        for line in self.transcription:
            print(line)

    def use_mic(self):
        if 'linux' in platform:
            mic_name = self.default_microphone
            if not mic_name or mic_name == 'list':
                print("Available microphone devices are: ")
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    print(f"Microphone with name \"{name}\" found")
                return
            else:
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    if mic_name in name:
                        self.source = sr.Microphone(sample_rate=16000, device_index=index)
                        break
        else:
            self.source = sr.Microphone(sample_rate=16000)

        with self.source as source:
            self.recorder.adjust_for_ambient_noise(source)
        
        def recording_callback(_, audio:sr.AudioData) -> None:
            data = audio.get_raw_data()
            self.data_queue.put(data)
            
        self.recorder.listen_in_background(self.source, recording_callback, phrase_time_limit=self.record_timeout)

def main():
    transcriber = Transcriber()
    transcriber.load_model()
    transcriber.use_mic()
    transcriber.execute()

if __name__ == "__main__":
    main()
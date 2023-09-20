import io
import os
import asyncio
import speech_recognition as sr
import whisper
import torch
import numpy as np
from scipy.io import wavfile

from datetime import datetime, timedelta
from queue import Queue
from sys import platform

class Transcriber:

    def __init__(self, model = "base", record_timeout = 2, phrase_timeout = 3) -> None:
        """
        Initialize the transcriber with the given parameters.

        Args:
            ``model`` (str, optional): The model to use. Defaults to "base". Choices are ["tiny", "base", "small", "medium", "large"].\n
            ``record_timeout`` (int, optional): How real time the recording is in seconds. Defaults to 2.\n
            ``phrase_timeout`` (int, optional): How much empty space between recordings before we consider it a new line in the transcription. Defaults to 3.\n
        """
        self.model = model
        self.record_timeout = record_timeout
        self.phrase_timeout = phrase_timeout
        self.phrase_complete = True

        # Current raw audio bytes
        self.last_sample = bytes()

        # Last time recording was retrieved from queue
        self.phrase_time = None

        # Thread safe Queue for passing data from the threaded recording callback.
        self.data_queue = Queue()
        self.data = sr.AudioData
        self.transcription = ['']

        self.load_model()

    def load_model(self):
        """
        Load the model into memory.
        """
        model = self.model
        
        if model != "large":
            model = model + ".en"

        self.loaded_model = whisper.load_model(model)

        # Signal the user that the model is ready
        print("Model loaded.")

    def execute(self, callback):
        asyncio.run(self.__execute__(callback))

    async def __execute__(self, callback):

        while True:
            try:
                # Time the audio was recieved from queue
                now = datetime.utcnow()

                # Pull raw audio from the queue
                if not self.data_queue.empty():

                    # If enough time has passed between recordings, consider phrase complete
                    # Clear working audio buffer to start over with new data
                    if self.phrase_time and now - self.phrase_time > timedelta(seconds=self.phrase_timeout):
                        self.last_sample = bytes()
                        self.phrase_complete = True
                    else:
                        self.phrase_complete = False

                    # This is the last time we received new audio data from the queue.
                    self.phrase_time = now

                    while not self.data_queue.empty():
                        self.data = self.data_queue.get()
                        self.last_sample += self.data

                    # convert raw to wav with AudioData
                    audio_data = sr.AudioData(self.last_sample, self.SAMPLE_RATE, self.SAMPLE_WIDTH)
                    wav_data = io.BytesIO(audio_data.get_wav_data())

                    # Set pointer to start of data
                    wav_data.seek(0)

                    # Read wav file from BytesIO object in mem
                    _, wav_data = wavfile.read(wav_data)

                    # Normalize audio & convert to numpy array
                    wav_data = (wav_data.astype(float) / np.iinfo(np.int16).max).astype(np.float32)

                    def infer():
                        # Transcribe audio
                        result = self.loaded_model.transcribe(wav_data, fp16=torch.cuda.is_available())
                        text = result['text'].strip()

                        if self.phrase_complete and text:
                            self.transcription.append(text)
                        elif text:
                            self.transcription[-1] = text

                        os.system('cls' if os.name=='nt' else 'clear')
                        for line in self.transcription:
                            print(line)
                        # Flush stdout.
                        print('', end='', flush=True)

                        # # Return result
                        #if not (len(text) == 0 or text.isspace()): callback(text)

                    loop = asyncio.get_event_loop()
                    #loop.run_in_executor(None, infer)

                    infer_future = loop.run_in_executor(None, infer)
                    await infer_future

                    # Reset sample
                    self.last_sample = self.last_sample[len(wav_data) - (len(wav_data) % self.SAMPLE_WIDTH):]

                    # Sleep for a bit to avoid CPU overload
                    await asyncio.sleep(0.15)
            except KeyboardInterrupt:
                break

    def use_mic(self, energy_threshold = 1000, default_microphone = "pulse"):
        """
        Callback for when audio is recieved.

        Args:
            ``energy_threshold`` (int, optional): The energy level for the microphone to detect. Defaults to 1000.\n
            ``default_microphone`` (str, optional): Only needed when running on linux machines! The default microphone to use. Defaults to "pulse".\n
        """

        recorder = sr.Recognizer()
        recorder.energy_threshold = energy_threshold

        # Make sure this is false, otherwise the energy threshold will be dynamic and lower to the point where it wont stop recording
        recorder.dynamic_energy_threshold = False

        # Important for linux machines
        if 'linux' in platform:
            mic_name = default_microphone
            if not mic_name or mic_name == 'list':
                print("Available microphone devices are: ")
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    print(f"Microphone with name \"{name}\" found")   
                return
            else:
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    if mic_name in name:
                        source = sr.Microphone(sample_rate=16000, device_index=index)
                        break
        else:
            source = sr.Microphone(sample_rate=16000)

        self.SAMPLE_RATE, self.SAMPLE_WIDTH = source.SAMPLE_RATE, source.SAMPLE_WIDTH

        with source:
            recorder.adjust_for_ambient_noise(source)

        def recording_callback(_, audio:sr.AudioData) -> None:
            """
            Threaded callback function to recieve audio data when recordings finish.
            audio: An AudioData containing the recorded bytes.
            """
            # Grab the raw bytes and push it into the thread safe queue.
            self.data = audio.get_raw_data()
            self.data_queue.put(self.data)
        
        # Create a background thread that will pass us raw audio bytes.
        # We could do this manually but SpeechRecognizer provides a nice helper.
        recorder.listen_in_background(source, recording_callback, phrase_time_limit=self.record_timeout)
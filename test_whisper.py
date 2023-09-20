from transcriber import Transcriber

# Set and load whisper model
transcriber = Transcriber() 

# You can use your mic after transcriber.use_mic() is called
transcriber.use_mic() 

def show(msg):
    pass

transcriber.execute(show)
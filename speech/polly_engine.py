import boto3
import numpy as np
import keyboard
import sounddevice as sd



class PollyEngine:
    def __init__(self):
        self.polly_client = boto3.client('polly')
        
    
    def speak(self, text):
        response = self.polly_client.synthesize_speech(
                Engine='standard',
                OutputFormat='pcm',
                Text=text,
                VoiceId='Matthew'
        )
            
        audio_bytes = response['AudioStream'].read()
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
    
        sd.play(audio_array, samplerate=13500)
        keyboard.add_hotkey('space',sd.stop)
        sd.wait()
            

    

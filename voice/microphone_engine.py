import queue
import sounddevice as sd
from collections import deque
import numpy as np

class MicrophoneEngine:

    def __init__(self,
                 sample_rate=16000,
                 channels=1,
                 blocksize=512):

        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize

        self.stream = sd.InputStream(
        samplerate=self.sample_rate,
        channels=self.channels,
        blocksize=self.blocksize,
        callback=self.audio_callback,
        dtype="float32",
    )
        self.audio_queue = queue.Queue(maxsize=50)
        self.ring_buffer = deque(maxlen=16000)
    
    def audio_callback(self, indata, frames, time, status):
        

        if status:
            print(status)

        try:
            self.audio_queue.put_nowait(indata.copy())
            self.ring_buffer.extend(indata.flatten())
        except queue.Full:
            # Drop the oldest chunk instead of blocking
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(indata.copy())
            except queue.Empty:
                pass
        
    def start(self):

        print("[Mic] Starting...")

        self.stream.start()

        print("[Mic] Ready")
    
    def stop(self):

        self.stream.stop()
        self.stream.close()
        
    def get_chunk(self):

        return self.audio_queue.get()
    
    def flush(self):

        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
    def get_buffer(self):
        return np.array(self.ring_buffer, dtype=np.float32)
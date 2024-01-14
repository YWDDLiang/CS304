import pyaudio
CHUNK = 1600
FORMAT = pyaudio.paInt16 
CHANNELS = 1 
RATE = 16000
Expected_duration=1
Expected_num_of_chunks= int(RATE/CHUNK*Expected_duration) 
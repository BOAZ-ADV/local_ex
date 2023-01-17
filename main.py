import streamlit as st
from audio_recorder_streamlit import audio_recorder
from io import BytesIO
from pydub import AudioSegment
import speech_recognition as sr
import STT
import time

st.title("Audio Recorder")



audio_bytes = audio_recorder("Click to record", pause_threshold=10.0)
if audio_bytes:
    print(len(audio_bytes))
    st.audio(audio_bytes, format="audio/wav")
    with open("audio.wav", "wb") as f:
        f.write(audio_bytes)
    # wav_file = open("audio.wav", "wb")
    # wav_file.write(audio_bytes)
    # wav_file.close()



# AudioSegment.from_raw(BytesIO(audio_bytes), sample_width=2, frame_rate=32000, channels=2).export('./testttt.wav', format='wav')
# wav_file = open("audio.wav", "wb")
# wav_file.write(audio_bytes.tobytes()).export('./asdf.wav', format = 'wav')

id = STT.BitoPost("audio.wav")
time.sleep(5)
result = STT.BitoGet(id)
st.markdown(f'{result}')





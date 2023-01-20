import streamlit as st
from audio_recorder_streamlit import audio_recorder
import plotly.figure_factory as ff
# from io import BytesIO
# from pydub import AudioSegment
# import speech_recognition as sr

import STT
from time import time, sleep
import pyaudio
import wave

text_data = []

def speech_to_text(DIR: str):
    id = STT.BitoPost(DIR)
    sleep(5)
    result = STT.BitoGet(id)
    return result




def main():
    st.title('👮보이스피싱 잡아라👮')
    _pyaudio = None
    _stream = None
    wav_file = None
    if st.button('start'):
        _pyaudio = pyaudio.PyAudio()
        _pyaudio.open(
            format=pyaudio.paInt16,
                channels=2,
                rate=44100,
                input=True,
                frames_per_buffer=1024
        )
        wav_file = wave.open('audio.wav', "wb")
        wav_file.setnchannels(2)
        wav_file.setsampwidth(_pyaudio.get_sample_size(pyaudio.paInt16))
        wav_file.setframerate(44100)
        duration = 5
        for _ in range(int(44100 * duration / 1024)):
            audio_data = _stream.read(1024)
        wav_file.writeframes(audio_data)
        text_result = speech_to_text("audio.wav")
        st.markdown(f'결과: {text_result}')

    
    # audio_bytes = audio_recorder("Click to record", pause_threshold=10.0)
    # if audio_bytes:
    #     with open("audio.wav", "wb") as f:
    #         f.write(audio_bytes)
    #     text_result = speech_to_text("audio.wav")
    #     text_data.append(text_result)
    #     st.markdown(f'결과: {text_result}')
    


if __name__ == "__main__":
    main()
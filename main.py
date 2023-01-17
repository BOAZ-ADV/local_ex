import streamlit as st
from audio_recorder_streamlit import audio_recorder
from io import BytesIO
from pydub import AudioSegment
import speech_recognition as sr

# import numpy as np
# from scipy.io.wavfile import write
# import soundfile as sf
# import wave
# import streamlit.components.v1 as components
# import os

# sound_chunk = pydub.AudioSegment.empty()

# for audio_frame in audio_bytes:
#     sound = pydub.AudioSegment(
#                         data=audio_frame.to_ndarray().tobytes(),
#                         sample_width=audio_frame.format.bytes,
#                         frame_rate=audio_frame.sample_rate,
#                         channels=len(audio_frame.layout.channels),
#                     )
#     sound_chunk += sound

# sound.export('./testttt.wav', format='wav')
st.title("Audio Recorder")


audio_bytes = audio_recorder("Click to record", pause_threshold=10.0)
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

AudioSegment.from_raw(BytesIO(audio_bytes), sample_width=2, frame_rate=32000, channels=2).export('./testttt.wav', format='wav')
# wav_file = open("audio.wav", "wb")
# wav_file.write(audio_bytes.tobytes()).export('./asdf.wav', format = 'wav')

r = sr.Recognizer()

test = sr.AudioFile('./testttt.wav')
with test as source:
    audio = r.record(source)


result = r.recognize_google(audio, language = 'ko-KR', show_all=True)['alternative']
rr = []
for i in result:
    rr.append(i['transcript'])
rr = '.'.join(rr)
st.markdown(f'{rr}')
# print(result['alternative'][0]['transcript'])
# print(result)

# data, samplerate = sf.read(io.BytesIO(audio_bytes))
# print(f'type1: {type(data)}')
# print(f'type2: {type(samplerate)}')

# parent_dir = os.path.dirname(os.path.abspath(__file__))
# build_dir = os.path.join(parent_dir, "st_audiorec/frontend/build")
# st_audiorec = components.declare_component("st_audiorec", path=build_dir)

# val = st_audiorec()
# st.write('Audio data received in the Python backend will appear below this message ...')
# print(val)
# if isinstance(val, dict):  # retrieve audio data
#     with st.spinner('retrieving audio-recording...'):
#         ind, val = zip(*val['arr'].items())
#         ind = np.array(ind, dtype=int)  # convert to np array
#         val = np.array(val)             # convert to np array
#         sorted_ints = val[ind]
#         stream = BytesIO(b"".join([int(v).to_bytes(1, "big") for v in sorted_ints]))
#         wav_bytes = stream.read()

#     # wav_bytes contains audio data in format to be further processed
#     # display audio data as received on the Python side
#     st.audio(wav_bytes, format='audio/wav')


import streamlit as st
from audio_recorder_streamlit import audio_recorder
import plotly.express as px
import joblib
# from io import BytesIO
from pydub import AudioSegment
# import speech_recognition as sr

import STT
from time import time, sleep
import pandas as pd
import math
# import librosa

text_data = []

def speech_to_text(DIR: str):
    id = STT.BitoPost(DIR)
    sleep(5)
    result = STT.BitoGet(id)
    return result

# def show_waveform(wav):
#     y, sr = librosa.load(wav, sr=16000)
#     time = np.linspace(0, len(y)/sr, len(y)) # time axis

#     fig, ax1 = plt.subplots() # plot
#     ax1.plot(time, y, color = 'b', label='speech waveform')
#     ax1.set_ylabel("Amplitude") # y 축
#     ax1.set_xlabel("Time [s]") # x 축
#     plt.title('wave form') # 제목
#     plt.show()

# def trim_audio_data(start, end, audio_file, save_file):
#     sr = 96000

#     y, sr = librosa.load(audio_file, sr=sr)

#     ny = y[sr*start :sr*end]

#     librosa.output.write_wav(save_file, ny, sr)

def chop_audio(file_path, segment_size):
    audio_data = AudioSegment.from_wav(file_path)
    chopped_audio = [x for x in audio_data[::segment_size]]
    return chopped_audio

def trim_audio(audio_data, second):
    second = 1
    seconds = second * 1000
    slice = []
    for i in range(int(math.ceil(len(audio_data)/seconds))):
        slice.append(audio_data[i*seconds:seconds*(i+1)])
    #     slice.export('newSong_{}.mp3'.format(i), format="mp3")
    return slice



def main():
    st.title('👮보이스피싱 잡아라👮')
    audio_bytes = audio_recorder("Click to record", pause_threshold=100.0)
    if audio_bytes:
        with open("audio.wav", "wb") as f:
            f.write(audio_bytes)
        # st.text('recording')
        # text_result = speech_to_text("audio.wav")
        # text_data.append(text_result)
        text_data = []
        result_dict = {}
        # chop_audios = chop_audio('audio.wav', 5)
        trim_audios = trim_audio('audio.wav', 5)
        for audio in trim_audios:
            # start = 0
            end = 5
            st.text('start')
            # trim_audio_data(start, end, "audio.wav", "cut_audio.wav")
            # audio.export('cut_audio.wav', format="wav")
            audio.export('cut_audio.wav', format="wav")
            st.text(audio)
            st.text('export')
            model = joblib.load('best_f1_model.pkl')
            encoder = joblib.load('best_tfvec.pkl')
            text_result = speech_to_text("cut_audio.wav")
            text_data.append(text_result)
            st.text(text_data)
            array = model.predict_proba(encoder.transform(text_data))
            prob = array[0][0]
            st.text(prob)
            result_dict[end] = prob
            # start += 5
            end += 5

        df = pd.DataFrame.from_dict([result_dict]).transpose().reset_index()
        df.columns = ['second', 'prob']
        fig = px.bar(pd.DataFrame(df), x='second', y='prob')
        
        tab1, tab2 = st.tabs(["output text", "plot"])
        with tab1:
            st.markdown(f'결과: {text_data}')
            # def show_waveform('audio.wav'):

        with tab2:
            st.plotly_chart(fig, theme=None)

    # fig = go.Figure(go.Scatter(x = list(result_dict.keys()),y = list(result_dict.values())))
    # st.plotly_chart(fig)


if __name__ == "__main__":
    main()
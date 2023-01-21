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



text_data = []

def speech_to_text(DIR: str):
    id = STT.BitoPost(DIR)
    sleep(5)
    result = STT.BitoGet(id)
    return result

# def show_waveframe(wav):



# def trim_audio_data(start, end, audio_file, save_file):
#     sr = 96000

#     y, sr = librosa.load(audio_file, sr=sr)

#     ny = y[sr*start :sr*end]

#     librosa.output.write_wav(save_file, ny, sr)

# def chop_audio(file_path, segment_size):
#     audio_data = AudioSegment.from_wav(file_path)
#     chopped_audio = [x for x in audio_data[::segment_size]]
#     return chopped_audio

# def trim_audio(audio_data, second):
#     second = 1
#     seconds = second * 1000
#     slice = []
#     for i in range(int(math.ceil(len(audio_data)/seconds))):
#         slice.append(audio_data[i*seconds:seconds*(i+1)])
#     #     slice.export('newSong_{}.mp3'.format(i), format="mp3")
#     return slice



def main():
    st.title('👮보이스피싱 잡아라👮')
    audio_bytes = audio_recorder(
    text="Click to record",
     pause_threshold=100.0,
    recording_color="#6aa36f",
    neutral_color="#6aa36f",
    icon_name="user",
    icon_size="3x",
)
    if audio_bytes:
        with open("audio.wav", "wb") as f:
            f.write(audio_bytes)
        # st.text('recording')
        # text_result = speech_to_text("audio.wav")
        # text_data.append(text_result)
        # text_data = []
        result_dict = {}
        text_data = speech_to_text("audio.wav")
        st.text('stt 진행 중 ')
        # st.text(text_data)

        st.text('call classification model & encoder')
        model = joblib.load('best_f1_model.pkl')
        encoder = joblib.load('best_tfvec.pkl')
        

        slice_num = 5
        for i in range(round(len(text_data)/slice_num)):
            text = text_data[ : slice_num*(1+i)]
            array = model.predict_proba(encoder.transform([text]))
            prob = array[0][0]
            st.text(prob)
            result_dict[slice_num*(1+i)] = 1 - prob #prob는 0에 가까울 수록 보이스 피싱임


        df = pd.DataFrame.from_dict([result_dict]).transpose().reset_index()
        df.columns = ['second', 'prob']
        fig = px.bar(pd.DataFrame(df), x='text_length', y='prob')
        
        tab1, tab2 = st.tabs(["output text", "plot"])
        with tab1:
            st.markdown(f'결과: {text_data}')

        with tab2:
            st.plotly_chart(fig, theme=None)



if __name__ == "__main__":
    main()
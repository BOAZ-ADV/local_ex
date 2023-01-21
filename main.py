import streamlit as st
from audio_recorder_streamlit import audio_recorder
import plotly.express as px
import joblib
# from io import BytesIO
# from pydub import AudioSegment
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




def main():
    st.title('👮보이스피싱 잡아라👮')
    audio_bytes = audio_recorder("Click to record", pause_threshold=100.0)
    if audio_bytes:
        # st.text('start')
        with open("audio.wav", "wb") as f:
            f.write(audio_bytes)
        # st.text('recording')
        text_result = speech_to_text("audio.wav")
        text_data.append(text_result)
    

        second = 0
        model = joblib.load('best_f1_model.pkl')
        encoder = joblib.load('best_tfvec.pkl')
        result_dict = {}
        array = model.predict_proba(encoder.transform(text_data))
        prob = array[0][0]
        second +=10
        result_dict[second] = prob

        df = pd.DataFrame.from_dict([result_dict]).transpose().reset_index()
        df.columns = ['second', 'prob']
        fig = px.bar(pd.DataFrame(df), x='second', y='prob')
        
        tab1, tab2 = st.tabs(["output text", "plot"])
        with tab1:
            st.markdown(f'결과: {text_data}')
        with tab2:
            st.plotly_chart(fig, theme=None)

    # fig = go.Figure(go.Scatter(x = list(result_dict.keys()),y = list(result_dict.values())))
    # st.plotly_chart(fig)


if __name__ == "__main__":
    main()
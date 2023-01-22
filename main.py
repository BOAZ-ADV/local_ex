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
from PIL import Image

st.set_page_config(layout="wide")
empty1,con1,con4,empty2 = st.columns([0.3,0.3,0.7,0.3])
empyt1,con2,con3,empty2 = st.columns([0.3,0.5,0.5,0.3])
# empyt1,con4,empty2 = st.columns([0.3,1.0,0.3])
# empyt1,con5,con6,empty2 = st.columns([0.3,0.5,0.5,0.3])


def speech_to_text(DIR: str):
    id = STT.BitoPost(DIR)
    sleep(5)
    result = STT.BitoGet(id)
    return result

def load_image(image_file):
    img = Image.open(image_file)
    return img

def main():
    # with empty1 : 
    #     empty()
    # with empty2:
    #     empty()
    # if 'text_data' not in st.session_state or 'prob' not in st.session_state or 'fig' not in st.session_state :
    #     st.session_state.text_data = ''
    #     st.session_state.prob= 0
    #     st.session_state.fig = 

    with con1:
        img = load_image('mask.png')
        st.image(img)
    with con4:
        st.title('👮보이스피싱 잡아라👮')
        audio_bytes = audio_recorder(
        text="Click to record",
        pause_threshold=100.0 # 100초 늘려야할듯..?
        # recording_color="#6aa36f",
        # neutral_color="#909090",
        # icon_name="volumne",
        # icon_size="3x",
    )
        if audio_bytes:
            with open("audio.wav", "wb") as f:
                f.write(audio_bytes)
            
            st.write('stt 진행 중')
            st.session_state.text_data = speech_to_text("audio.wav")


            st.write('call classification model & encoder')
            model = joblib.load('best_f1_model.pkl')
            encoder = joblib.load('best_tfvec.pkl')
            
            result_dict = {0:0}
            slice_num = 5 #slice 할 글자 수
            for i in range(round(len(st.session_state.text_data)/slice_num)):
                text = st.session_state.text_data[ : slice_num*(1+i)]
                array = model.predict_proba(encoder.transform([text]))
                st.session_state.prob = array[0][0]
                result_dict[slice_num*(1+i)] = 1 - st.session_state.prob #prob는 0에 가까울 수록 보이스 피싱임?


            df = pd.DataFrame.from_dict([result_dict]).transpose().reset_index()
            df.columns = ['text_length', 'prob']
            st.session_state.fig = px.area(df, x='text_length', y='prob', markers = True) #축 0~1로 고정하기
            
            # tab1, tab2 = st.tabs(["output text", "plot"])
        with con2:
            st.subheader('결과보기 🔽')
            if audio_bytes:
                audio_file = open("audio.wav", 'rb')
                st.audio( audio_file.read() , format='audio/wav')

                st.markdown(f'결과: {st.session_state.text_data}')
                st.text(round(1-st.session_state.prob,2))

        with con3:
            st.subheader('Chart📈')
            if audio_bytes:
                st.plotly_chart(st.session_state.fig, theme=None)



if __name__ == "__main__":
    main()
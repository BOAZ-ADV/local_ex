import streamlit as st
from audio_recorder_streamlit import audio_recorder
import plotly.express as px
import plotly.graph_objects as go
import joblib
# from io import BytesIO
# from pydub import AudioSegment
# import speech_recognition as sr

import STT
from time import time, sleep
import pandas as pd
from PIL import Image
# import streamlit_nested_layout

st.set_page_config(layout="wide")
empty1,con1,empty2 = st.columns([0.3,1,0.3])
empty1,con3,empty2 = st.columns([0.3,1,0.3])
empty1,con4,empty2 = st.columns([0.3,1,0.3])
empyt1,con5,empty2 = st.columns([0.3,1,0.3])
empty1,con6,empty2 = st.columns([0.3,1,0.3])
empyt1,con7,con8,empty2 = st.columns([0.3,0.3,0.7,0.3])
empyt1,con9,empty2 = st.columns([0.3,1,0.3])
empyt1,con10,empty2 = st.columns([0.3,1,0.3])



def speech_to_text(DIR: str):
    id = STT.BitoPost(DIR)
    sleep(5)
    result = STT.BitoGet(id)
    return result

def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


# @st.cache
def main():
    # with empty1 : 
    #     empty()
    # with empty2:
    #     empty()
    # local_css("style.css")
    with con1:
        img = Image.open('title.png')
        st.image(img)

    with con3:
        st.title('Voice Phishing Detection Algorithm 🔍')
        # st.write("[![Star](<https://img.shields.io/github/stars/><BOAZ-ADV>/<local_ex>.svg?logo=github&style=social)](<https://gitHub.com/><BOAZ-ADV>/<local_ex>)") #깃헙 repo 링크 변경하기
    with con4:
        st.subheader('🔴 Click to record ')
        audio_bytes = audio_recorder(
        text="",
        pause_threshold=100.0, # 100초 늘려야할듯..?
        # recording_color="#6aa36f",
        # neutral_color="#909090",
        # icon_name="volumne",
        icon_size="4x"
    )
    with con5:
        st.subheader('💡 Progress')
        if audio_bytes:
            st.markdown('⏹ **Stop Recording**')
            with open("audio.wav", "wb") as f:
                f.write(audio_bytes)
            st.markdown('⏳ **Speech To Text 진행 중...**')
            try:
                st.session_state.text_data = speech_to_text("audio.wav")
            except:
                st.markdown('❗ STT 변환 실패 다시 녹음하세요')

            st.markdown('🔧 **Call Classification Model & Encoder**')
            model = joblib.load('best_f1_model.pkl')
            encoder = joblib.load('best_tfvec.pkl')
            
            result_dict = {0:0}
            slice_num = 10 #slice 할 글자 수
            for i in range(round(len(st.session_state.text_data)/slice_num)):
                text = st.session_state.text_data[ : slice_num*(1+i)]
                array = model.predict_proba(encoder.transform([text]))
                st.session_state.prob = array[0][0]
                result_dict[slice_num*(1+i)] = 1 - st.session_state.prob #prob는 0에 가까울 수록 보이스 피싱임?
            st.markdown('🍀 **Finish**')

            df = pd.DataFrame.from_dict([result_dict]).transpose().reset_index()
            df.columns = ['Text Length', 'Voice Phishing Probabilty']
            st.session_state.fig = px.area(df, x='Text Length', y='Voice Phishing Probabilty', markers = True) #축 0~1로 고정하기
            # st.session_state.fig = go.Figure()
            # st.session_state.fig.add_trace(go.scatter(x=list(result_dict.keys()), y=list(result_dict.values()), mode = 'lines+markers'))
            st.session_state.fig.update_layout(paper_bgcolor = "white")
            # st.session_state.fig.update_layout(plot_bgcolor = "white")
            st.session_state.fig.update_yaxes(range=[0,1])
            # st.session_state.fig.update_layout(go.Layout(title={'text' : 'Vocie Phishing Probability',
            #                                                     'font':{'color':'black', 'size':25}},
            #                                     paper_bgcolor='#f8ec9c'))
            
            # tab1, tab2 = st.tabs(["output text", "plot"])
        with con6:
            st.subheader('📝 결과보기 ')

        if audio_bytes:
            with con7:
                result_prob = round(1-st.session_state.prob,3)
                if result_prob > 0.7:
                    st.image(Image.open('red.png'), width = 100)
                elif result_prob > 0.3:
                    st.image(Image.open('yellow.png'), width = 100)
                else:
                    st.image(Image.open('green.png'), width = 100)
            with con8:
                result_prob = round(1-st.session_state.prob,3)
                # st.title(f'{result_prob*100}%')
                if result_prob > 0.7:
                    st.subheader(f"보이스피싱 확률이 {result_prob*100}% 으로 위험")
                elif result_prob > 0.3:
                    st.title(f"보이스피싱 확률이 {result_prob*100}% 으로 경고")
                else:        
                    st.title(f"보이스피싱 확률이 {result_prob*100}% 으로 안전")

                    # st.markdown("""
                    # <style>
                    # .big-font {font-size:70px ;}
                    # </style>
                    # """, unsafe_allow_html=True)
                    # st.markdown(f'<p class="big-font">{result_prob*100}%</p>', unsafe_allow_html=True)
                    
            with con9:
                audio_file = open("audio.wav", 'rb')
                st.audio( audio_file.read() , format='audio/wav')
                # local_css("style.css")
                with st.expander('📂 RESULT TEXT', expanded=True):
                    st.markdown(st.session_state.text_data)


        with con10:
            st.subheader('📊 Chart')
            if audio_bytes:
                st.plotly_chart(st.session_state.fig, theme = "streamlit")



if __name__ == "__main__":
    main()
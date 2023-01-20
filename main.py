import streamlit as st
from audio_recorder_streamlit import audio_recorder
# import plotly.figure_factory as ff
# from io import BytesIO
# from pydub import AudioSegment
# import speech_recognition as sr

import STT
from time import time, sleep
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from pydub import AudioSegment
import queue
from streamlit.state.session_state import LazySessionState

text_data = []

def speech_to_text(DIR: str):
    id = STT.BitoPost(DIR)
    sleep(5)
    result = STT.BitoGet(id)
    return result




def main():
    st.title('👮보이스피싱 잡아라👮')
    with st.container():
        sample = st.session_state.audio_buffer
        audio_available = sample != AudioSegment.empty()
        if audio_available:
            st.audio(
                sample.export(format="wav", codec="pcm_s16le", bitrate="128k").read()
            )
        else:
            with (record_section := st.container()):
                webrtc_ctx = webrtc_streamer(
                    key="sendonly-audio",
                    mode=WebRtcMode.SENDONLY,
                    audio_receiver_size=1024,
                    rtc_configuration={
                        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                    },
                    media_stream_constraints={"audio": True, "video": False},
                )

                with st.spinner(text="recording..."):
                    while True:
                        if webrtc_ctx.audio_receiver:
                            try:
                                audio_frames = webrtc_ctx.audio_receiver.get_frames(
                                    timeout=3
                                )
                            except queue.Empty:
                                record_section.write("no audio received...")
                            sound_chunk = AudioSegment.empty()
                            try:
                                for audio_frame in audio_frames:
                                    sound = AudioSegment(
                                        data=audio_frame.to_ndarray().tobytes(),
                                        sample_width=audio_frame.format.bytes,
                                        frame_rate=audio_frame.sample_rate,
                                        channels=len(audio_frame.layout.channels),
                                    )
                                    sound_chunk += sound
                                if len(sound_chunk) > 0:
                                    session_state.audio_buffer += sound_chunk
                            except UnboundLocalError:
                                # UnboundLocalError when audio_frames is not set
                                record_section.write("no audio detected...")
                        else:
                            break

    
    # audio_bytes = audio_recorder("Click to record", pause_threshold=10.0)
    # if audio_bytes:
    #     with open("audio.wav", "wb") as f:
    #         f.write(audio_bytes)
    #     text_result = speech_to_text("audio.wav")
    #     text_data.append(text_result)
    #     st.markdown(f'결과: {text_result}')
    


if __name__ == "__main__":
    main()
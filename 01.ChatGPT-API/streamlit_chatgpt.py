"""
아래 코드를 ***.py로 저장하고, streamlit run ***.py로 실행

!pip install streamlit
"""

import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

st.title("Streamlit ChatGPT")

if 'messages' not in st.session_state:
    st.session_state.messages = []

user_message = st.text_input("User:")

if st.button("Send"):
    st.session_state.messages.append({"role": "user", "content": user_message})

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=st.session_state.messages,
    )
    response = completion.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": response})

for message in reversed(st.session_state.messages):
    with st.chat_message(message['role']):
        if message['role'] == 'user':
            st.write(message['content'])
        else:
            st.write(message['content'], avatar=st.image('assets/openai-logo.webp', width=30))
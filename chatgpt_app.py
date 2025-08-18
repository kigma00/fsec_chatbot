# pip install openai==1.20.0 streamlit
import streamlit as st
from openai import OpenAI

api_key = "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"
client = OpenAI(api_key=api_key)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.chat_message("assistant"):
    st.write("안녕하세요! 뉴스기사를 한 문장으로 요약해드릴게요!")

prompt = st.chat_input("번역할 문장")

if prompt:
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "너는 뉴스기사 요약 전문 챗봇이야."},
        {"role": "user", "content": f"뉴스기사\n{prompt}\n\n위 뉴스기사를 한 문장으로 요약해줘."}
        ]
    )
    st.session_state.messages.append((prompt, response.choices[0].message.content))

for prompt, answer in st.session_state.messages:
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        st.write(answer) 
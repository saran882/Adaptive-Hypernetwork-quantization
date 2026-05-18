import streamlit as st
import ollama
st.title("Chat with Llama 3.1")
prompts = st.text_input("Enter your prompt:")
button = st.button("Submit")
if button:
    if prompts:
        response=ollama.generate(model="llama3.1",prompt= prompts)
        st.markdown(response["response"])

    # client = Client()
    # response = client.chat('llama3.1', messages=[{"role": "user", "content": prompts}])
    # st.write(response['message']['content'])
import streamlit as st
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

st.title("⚖️ Legal Expert Q&A")
st.markdown("Ask general legal questions. *Note: This bot does not have access to your uploaded files.*")

# Initialize Chat History
if "qa_messages" not in st.session_state:
    st.session_state.qa_messages = [
        {"role": "assistant", "content": "I am your AI legal assistant. How can I help with your legal research today?"}
    ]

# Display History
for msg in st.session_state.qa_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Ex: What are the elements of a valid contract in the UK?"):
    st.session_state.qa_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Reasoning through legal principles..."):
            llm = ChatGroq(
                model="deepseek-r1-distill-llama-70b", 
                temperature=0.1, # For more natural legal drafting
                groq_api_key=os.getenv("GROQ_API_KEY")
            )
            
            # System instruction to act as a lawyer
            system_prompt = (
                "You are a senior legal expert. Provide detailed, structured, and formal legal information. "
                "Always include relevant legal principles and suggest consulting a qualified human attorney."
            )
            
            response = llm.invoke(f"{system_prompt}\n\nUser Question: {prompt}")
            st.markdown(response.content)
            st.session_state.qa_messages.append({"role": "assistant", "content": response.content})
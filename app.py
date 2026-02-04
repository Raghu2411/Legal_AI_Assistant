# import streamlit as st
# import os
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_chroma import Chroma
# from langchain_community.document_loaders import UnstructuredFileLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.chains.retrieval_qa.base import RetrievalQA
# from dotenv import load_dotenv

# load_dotenv()

# # --- UI Setup ---
# st.set_page_config(page_title="Legal AI Analyst", layout="wide")
# st.title("⚖️ Legal Document AI Analyst")
# st.markdown("Upload a legal PDF/DOCX to extract clauses, summarize, or explain.")

# with st.sidebar:
#     st.header("Upload Document")
#     uploaded_file = st.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"])
#     process_button = st.button("Analyze Document")

# # --- Logic ---
# if uploaded_file and process_button:
#     with st.spinner("Processing legal document..."):
#         # 1. Save file locally
#         temp_path = f"temp_{uploaded_file.name}"
#         with open(temp_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())

#         # 2. Load and Split
#         loader = UnstructuredFileLoader(temp_path)
#         docs = loader.load()
        
#         # Legal docs need smaller chunks to keep clause context together
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
#         splits = text_splitter.split_documents(docs)

#         # 3. Create Vector Store
#         vectorstore = Chroma.from_documents(
#             documents=splits, 
#             embedding=OpenAIEmbeddings()
#         )
        
#         # 4. Initialize Retrieval Chain
#         llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
#         st.session_state.qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff",
#             retriever=vectorstore.as_retriever(),
#             return_source_documents=True
#         )
#         st.success("Document Indexed! You can now ask questions.")

# # --- Chat Interface ---
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat history
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Chat Input
# if prompt := st.chat_input("Ex: 'Summarize the termination clause' or 'Are there any non-competes?'"):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     if "qa_chain" in st.session_state:
#         with st.chat_message("assistant"):
#             result = st.session_state.qa_chain.invoke({"query": prompt})
#             answer = result["result"]
#             sources = result["source_documents"]
            
#             st.markdown(answer)
            
#             # Show Citations for legal transparency
#             with st.expander("View Source Passages"):
#                 for i, doc in enumerate(sources):
#                     st.info(f"Source {i+1}: {doc.page_content[:300]}...")
            
#             st.session_state.messages.append({"role": "assistant", "content": answer})
#     else:
#         st.error("Please upload and process a document first!")

import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

# --- UI Setup ---
st.set_page_config(page_title="Legal AI Analyst", layout="wide")
st.title("⚖️ Legal Document AI Analyst")
st.markdown("Upload a legal PDF/DOCX to extract clauses, summarize, or explain.")

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"])
    process_button = st.button("Analyze Document")

# --- Logic ---
if uploaded_file and process_button:
    with st.spinner("Processing legal document..."):
        # 1. Save file locally
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 2. Load and Split
        loader = UnstructuredFileLoader(temp_path)
        docs = loader.load()
        
        # Legal docs need smaller chunks to keep clause context together
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)

        # 3. Create Vector Store with HuggingFace embeddings (free, local)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings
        )
        
        # 4. Initialize Retrieval Chain with Groq
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",  # or "mixtral-8x7b-32768" for faster responses
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )
        st.success("Document Indexed! You can now ask questions.")

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ex: 'Summarize the termination clause' or 'Are there any non-competes?'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "qa_chain" in st.session_state:
        with st.chat_message("assistant"):
            result = st.session_state.qa_chain.invoke({"query": prompt})
            answer = result["result"]
            sources = result["source_documents"]
            
            st.markdown(answer)
            
            # Show Citations for legal transparency
            with st.expander("View Source Passages"):
                for i, doc in enumerate(sources):
                    st.info(f"Source {i+1}: {doc.page_content[:300]}...")
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.error("Please upload and process a document first!")
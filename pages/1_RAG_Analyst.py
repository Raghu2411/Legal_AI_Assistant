import streamlit as st
import os
import hashlib
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

# UI Setup
st.set_page_config(page_title="Legal AI Analyst", layout="wide")
st.title("⚖️ Legal Document AI Analyst")
st.markdown("Upload multiple documents to test extraction and analysis.")

# Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("📁 Document Management")
    
    if "current_doc" in st.session_state:
        st.success(f"📄 Active: **{st.session_state.current_doc}**")
    
    st.divider()
    
    uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])
    process_button = st.button("Analyze & Switch Context")
    
    st.divider()
    st.info("💡 Each time you click 'Analyze', the AI switches its focus to the new document.")


if uploaded_file and process_button:
    # Clear previous session messages
    st.session_state.messages = []
    
    with st.spinner(f"Processing {uploaded_file.name}..."):
        storage_dir = "./chromadb_storage"
        os.makedirs(storage_dir, exist_ok=True)
        
        # New unique path for this file
        file_hash = hashlib.md5(uploaded_file.name.encode()).hexdigest()[:8]
        persist_directory = os.path.join(storage_dir, f"chroma_db_{file_hash}")
        
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            # Document Processing Pipeline
            loader = UnstructuredFileLoader(temp_path)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=150
            )
            splits = text_splitter.split_documents(docs)

            # Embedding & Vector Store
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            vectorstore = Chroma.from_documents(
                documents=splits, 
                embedding=embeddings,
                persist_directory=persist_directory
            )

            # Retrieval Chain Setup
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0,
                groq_api_key=os.getenv("GROQ_API_KEY")
            )
            
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )
            
            st.session_state.current_doc = uploaded_file.name
            st.success(f"Ready to analyze: {uploaded_file.name}")
            
        except Exception as e:
            st.error(f"Error processing document: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

# Chat Interface
if "qa_chain" in st.session_state:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input(f"Ask about {st.session_state.current_doc}..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Reviewing document..."):
                result = st.session_state.qa_chain.invoke({"query": prompt})
                answer = result["result"]
                sources = result["source_documents"]
                
                st.markdown(answer)
                
                with st.expander("View Legal Citations"):
                    for i, doc in enumerate(sources):
                        st.info(f"Reference {i+1}: {doc.page_content[:400]}...")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("👋 Welcome! Please upload a legal document in the sidebar to begin testing.")
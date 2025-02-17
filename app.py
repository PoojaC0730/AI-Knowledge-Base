import streamlit as st
import os
import pandas as pd
from datetime import datetime
import json
import base64
from typing import List, Dict
import tempfile
import uuid
import hashlib

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, LLMChain
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Set HuggingFace API token
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")  # Load token from environment variable

class KnowledgeBase:
    def __init__(self):
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            persist_directory="./knowledge_base",
            embedding_function=self.embeddings
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Initialize LLM
        self.llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",
            huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
            model_kwargs={"temperature": 0.5, "max_length": 512}
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory()
        
        # Create QA Chain
        self.qa_chain = self._create_qa_chain()
        
        # Create Chat Chain
        self.chat_chain = self._create_chat_chain()
        
        # Create Summarization Chain
        self.summary_chain = self._create_summary_chain()
        
        # Track processed files
        self.processed_files = set()

    def _create_qa_chain(self):
        prompt_template = """
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        
        Previous conversation:
        {history}
        
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question", "history"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT, "memory": self.memory}
        )

    def _create_chat_chain(self):
        chat_template = """
        You are a helpful AI assistant with access to a knowledge base of documents and notes.
        Use the following conversation history and your knowledge to provide a helpful response.

        Current conversation:
        {history}
        Human: {input}
        Assistant:"""
        
        CHAT_PROMPT = PromptTemplate(
            template=chat_template,
            input_variables=["history", "input"]
        )
        
        return ConversationChain(
            llm=self.llm,
            prompt=CHAT_PROMPT,
            memory=self.memory,
            verbose=True
        )

    def _create_summary_chain(self):
        summary_template = """
        Please provide a concise summary of the following text:
        
        {text}
        
        Summary:"""
        
        SUMMARY_PROMPT = PromptTemplate(
            template=summary_template,
            input_variables=["text"]
        )
        
        return LLMChain(
            llm=self.llm,
            prompt=SUMMARY_PROMPT
        )

    def process_pdf(self, pdf_path: str, metadata: Dict) -> tuple:
        try:
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Generate summary
            full_text = "\n".join([doc.page_content for doc in documents])
            summary = self.summary_chain.run(text=full_text)
            
            # Add metadata to documents
            for doc in documents:
                doc.metadata.update(metadata)
            
            # Split documents
            splits = self.text_splitter.split_documents(documents)
            
            # Add to vector store
            self.vector_store.add_documents(splits)
            
            return str(uuid.uuid4()), summary
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")

    def add_note(self, note: str, metadata: Dict) -> str:
        # Split note into chunks
        texts = self.text_splitter.split_text(note)
        
        # Create documents with metadata
        from langchain.schema import Document
        documents = [
            Document(
                page_content=text,
                metadata={**metadata, "chunk_id": i}
            )
            for i, text in enumerate(texts)
        ]
        
        # Add to vector store
        self.vector_store.add_documents(documents)
        
        return str(uuid.uuid4())

    def chat(self, message: str) -> tuple:
        try:
            # First try to get relevant context
            relevant_docs = self.vector_store.similarity_search(message, k=2)
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            # Combine context with the message
            if context:
                full_prompt = f"Context: {context}\n\nUser message: {message}"
            else:
                full_prompt = message
            
            # Get response using chat chain
            response = self.chat_chain.predict(input=full_prompt)
            
            return response, relevant_docs
        except Exception as e:
            return f"Error processing your request: {str(e)}", []

def create_chat_message(is_user: bool, message: str, key: str):
    if is_user:
        st.write(
            f'<div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">'
            f'<div style="background-color: #007AFF; color: white; padding: 10px; '
            f'border-radius: 10px; max-width: 80%;">{message}</div></div>', 
            unsafe_allow_html=True
        )
    else:
        st.write(
            f'<div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">'
            f'<div style="background-color: #E9ECEF; padding: 10px; '
            f'border-radius: 10px; max-width: 80%;">{message}</div></div>', 
            unsafe_allow_html=True
        )

def calculate_file_hash(file_content: bytes) -> str:
    """Calculate SHA-256 hash of file content."""
    return hashlib.sha256(file_content).hexdigest()

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to disk and return the file path."""
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return file_path

def get_pdf_download_link(pdf_path: str, filename: str) -> str:
    """Generate a download link for PDF files."""
    with open(pdf_path, "rb") as f:
        bytes_data = f.read()
    b64 = base64.b64encode(bytes_data).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def load_notes() -> Dict:
    """Load notes from the JSON file."""
    notes_file = "notes.json"
    if not os.path.exists(notes_file):
        # Create empty notes file if it doesn't exist
        save_notes({})
        return {}
    try:
        with open(notes_file, "r", encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        # Return empty dict if file is corrupted
        return {}

def save_notes(notes: Dict) -> None:
    """Save notes to the JSON file."""
    notes_file = "notes.json"
    try:
        with open(notes_file, "w", encoding='utf-8') as f:
            json.dump(notes, f, ensure_ascii=False, indent=4)
    except Exception as e:
        st.error(f"Error saving notes: {str(e)}")

def clear_chat_history():
    """Clear the chat history and reset the knowledge base memory."""
    st.session_state.chat_history = []
    st.session_state.knowledge_base.memory.clear()

def initialize_session_state():
    """Initialize all session state variables."""
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = KnowledgeBase()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'notes' not in st.session_state:
        st.session_state.notes = load_notes()
        
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()

def main():
    st.set_page_config(layout="wide", page_title="Personal Knowledge Base")
    
    # Initialize session state
    initialize_session_state()
    
    st.title("📚 Personal AI Knowledge Base")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📝 Notes", "📄 Documents", "📊 Knowledge Base"])
    
    # Tab 1: Chat Interface
    with tab1:
        st.header("Chat with Your Knowledge Base")
    
        # Add clear chat button
        if st.button("Clear Chat"):
            clear_chat_history()
            st.rerun()

        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                create_chat_message(
                    message["is_user"],
                    message["message"],
                    str(message["timestamp"])
                )
    
        # Chat input with send button
        with st.container():
            col1, col2 = st.columns([4, 1])
        
            # Initialize the chat input key in session state if it doesn't exist
            if "chat_input_key" not in st.session_state:
                st.session_state.chat_input_key = 0
        
            with col1:
                # Use a dynamic key for the text input
                chat_input = st.text_input(
                    "Type your message:", 
                    key=f"chat_input_{st.session_state.chat_input_key}"
                )
            with col2:
                send_message = st.button("Send")
            
            if send_message and chat_input:
                # Add user message to history
                st.session_state.chat_history.append({
                    "is_user": True,
                    "message": chat_input,
                    "timestamp": datetime.now()
                })
            
                # Get AI response
                with st.spinner("Thinking..."):
                    response, sources = st.session_state.knowledge_base.chat(chat_input)

                    # Add AI response to history
                    st.session_state.chat_history.append({
                        "is_user": False,
                        "message": response,
                        "timestamp": datetime.now()
                    })
            
                # Increment the key to force a new text input widget
                st.session_state.chat_input_key += 1

                # Force refresh
                st.rerun()
    
    # Tab 2: Notes
    with tab2:
        st.header("Notes")
        
        # Note creation
        with st.form("new_note"):
            note_title = st.text_input("Note Title")
            note_content = st.text_area("Note Content")
            submit_note = st.form_submit_button("Save Note")
            
            if submit_note and note_title and note_content:
                metadata = {
                    "title": note_title,
                    "source_type": "note",
                    "timestamp": datetime.now().isoformat()
                }
                note_id = st.session_state.knowledge_base.add_note(note_content, metadata)
                st.session_state.notes[note_id] = {
                    "title": note_title,
                    "content": note_content,
                    "timestamp": datetime.now().isoformat()
                }
                save_notes(st.session_state.notes)
                st.success("Note saved successfully!")
        
        # Display existing notes
        if st.session_state.notes:
            st.subheader("Your Notes")
            for note_id, note_data in st.session_state.notes.items():
                with st.expander(f"📝 {note_data['title']}"):
                    st.write(note_data['content'])
                    st.caption(f"Created: {note_data['timestamp']}")
    
    # Tab 3: Documents
    with tab3:
        st.header("Document Management")
        
        uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")
        if uploaded_file:
            file_content = uploaded_file.getvalue()
            file_hash = calculate_file_hash(file_content)
            
            # Check if file was already processed
            if file_hash not in st.session_state.processed_files:
                if st.button("Process Document"):
                    try:
                        with st.spinner("Processing document..."):
                            file_path = save_uploaded_file(uploaded_file)
                            metadata = {
                                "filename": uploaded_file.name,
                                "file_path": file_path,
                                "source_type": "pdf",
                                "upload_time": datetime.now().isoformat()
                            }
                            doc_id, summary = st.session_state.knowledge_base.process_pdf(file_path, metadata)
                            
                            # Mark file as processed
                            st.session_state.processed_files.add(file_hash)
                            
                            # Display summary
                            st.success("Document processed successfully!")
                            with st.expander("Document Summary"):
                                st.write(summary)
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
            else:
                st.info("This document has already been processed.")
        
        # Display uploaded documents
        if os.path.exists("uploads"):
            st.subheader("Uploaded Documents")
            for filename in os.listdir("uploads"):
                if filename.endswith(".pdf"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"📄 {filename}")
                    with col2:
                        st.markdown(get_pdf_download_link(
                            os.path.join("uploads", filename),
                            filename
                        ), unsafe_allow_html=True)
    # Tab 4: Knowledge Base Stats
    with tab4:
        st.header("Knowledge Base Statistics")
        
        total_notes = len(st.session_state.notes)
        total_docs = len(os.listdir("uploads")) if os.path.exists("uploads") else 0
        total_processed = len(st.session_state.processed_files)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Notes", total_notes)
        with col2:
            st.metric("Total Documents", total_docs)
        with col3:
            st.metric("Processed Documents", total_processed)

if __name__ == "__main__":
    main()
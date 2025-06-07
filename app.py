import streamlit as st
import yt_dlp
import whisper
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import tempfile
import os

st.title("YouTube Video Q&A with RAG")

openai_api_key = os.getenv("OPENAI_API_KEY")
whisper_model = whisper.load_model("base")

youtube_url = st.text_input("Enter YouTube URL")
query = st.text_input("Ask a question about the video")

if st.button("Process Video"):
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': audio_path,
            'quiet': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        
        result = whisper_model.transcribe(audio_path)
        transcript = result['text']
        st.write("Transcript:", transcript[:500] + "...")
        
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(transcript)
        docs = [Document(page_content=chunk) for chunk in chunks]

        embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectordb = FAISS.from_documents(docs, embedding)

        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(openai_api_key=openai_api_key),
            retriever=vectordb.as_retriever(),
            return_source_documents=True
        )
        
        st.session_state['qa'] = qa  

if query and 'qa' in st.session_state:
    answer = st.session_state['qa'].run(query)
    st.write("Answer:", answer)

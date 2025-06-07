# youtube-rag-searcher
Ask anything about a YouTube video using AI — powered by Whisper, LangChain, and OpenAI.

# YouTube Video Q&A with RAG (Retrieval-Augmented Generation)

This app lets you ask questions about any YouTube video by automatically downloading the audio, transcribing it using Whisper, creating vector embeddings, and answering queries with a retrieval-based LLM approach.

Built with:  
- **Streamlit** for the interactive web UI  
- **yt-dlp** to download YouTube audio  
- **OpenAI Whisper** for transcription  
- **LangChain** + **FAISS** for document retrieval and embeddings  
- **OpenAI GPT** for answering questions

---

## Features

- Input a YouTube video URL and get an automatic transcript summary  
- Ask follow-up questions based on the video content  
- Fast, easy-to-use chatbot-like interface  
- No manual transcript or indexing needed  

---



### Prerequisites

- Python 3.8 or higher  
- [FFmpeg](https://ffmpeg.org/) installed and in your system PATH (required for audio extraction)  
- OpenAI API key (set as environment variable `OPENAI_API_KEY`)


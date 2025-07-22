import streamlit as st
import os
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.schema.runnable import RunnableLambda
import re

# --- HELPER FUNCTIONS (Your RAG Core Logic) ---

def get_video_id(url):
    """Extracts the video ID from a YouTube URL."""
    # Regex to find video ID in various YouTube URL formats
    match = re.search(r"(?<=v=)[^&#]+|(?<=be/)[^&#]+|(?<=embed/)[^&#]+|(?<=shorts/)[^&#]+", url)
    if match:
        return match.group(0)
    # Fallback for googleusercontent URLs
    if 'googleusercontent.com/youtube.com' in url:
        return url.split('/')[-1].split('?')[0]
    return None

def get_transcript_text(url):
    """
    Fetches the transcript for a YouTube video using youtube_transcript_api.
    """
    try:
        video_id = get_video_id(url)
        if not video_id:
            return None, "Could not extract video ID from the URL."
            
        # ytt_api = YouTubeTranscriptApi()
        # transcript_list = ytt_api.get_transcript(video_id)
        ytt_api = YouTubeTranscriptApi()
        transcript_list=ytt_api.fetch(video_id)
        
        # Combine transcript text parts into a single string
        transcripts_new = [item.text for item in transcript_list]
        full_transcript = " ".join(transcripts_new)
        
        return full_transcript, None
    except Exception as e:
        return None, f"An error occurred while fetching the transcript: {e}"


def get_vector_store(text, gemini_api_key):
    """
    Creates and returns a FAISS vector store from the given text.
    """
    if not text:
        return None

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100
    )
    chunks = text_splitter.create_documents([text])

    # Create embeddings and vector store
    # Note: Using a local model for embeddings is great for privacy and cost.
    emb_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, emb_model)
    
    return vectorstore

def create_rag_chain(vectorstore, gemini_api_key):
    """
    Creates and returns a LangChain RAG chain.
    """
    # Set up the LLM
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', google_api_key=gemini_api_key)

    # Create the retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Using Maximal Marginal Relevance for diverse results
        search_kwargs={"k": 10}
    )

    # Define the prompt template
    prompt = PromptTemplate(
        template="""
        Assume you are a helpful YouTube Q&A Chat Bot.
        A user has asked the following question:
        '{query}'

        Here is the context from the video transcript:
        '{context}'

        Your task is to provide a clear and summarized answer to the user's query based ONLY on the provided transcript context. Do not use any external knowledge. If the context doesn't contain the answer, say so.
        """,
        input_variables=['query', 'context']
    )
    
    # Define the RAG chain
    def get_context(query):
        return retriever.invoke(query)

    rag_chain = (
        {
            "context": RunnableLambda(lambda x: x['query']) | retriever | format_docs,
            "query": RunnableLambda(lambda x: x['query'])
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain
def format_docs(docs):
    """Combines document contents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# --- STREAMLIT UI ---

st.set_page_config(page_title="Chat with YouTube", page_icon="📺", layout="wide")

st.title("📺 Chat with any YouTube Video")
st.markdown("This app lets you chat with a YouTube video. Just paste the URL, provide your Gemini API key, and start asking questions!")

# --- SIDEBAR for Inputs ---
with st.sidebar:
    st.header("Setup")
    
    youtube_url = st.text_input("Enter YouTube URL:", key="youtube_url_input")
    gemini_api_key = st.text_input("Enter your Gemini API Key:", type="password", key="gemini_api_key_input")

    if st.button("Process Video", key="process_button"):
        if not youtube_url:
            st.error("Please enter a YouTube URL.")
        elif not gemini_api_key:
            st.error("Please enter your Gemini API Key.")
        else:
            with st.spinner("Processing video... This may take a moment."):
                # Set API key in environment
                os.environ["GOOGLE_API_KEY"] = gemini_api_key
                
                # Fetch transcript
                transcript, error_message = get_transcript_text(youtube_url)
                
                if error_message:
                    st.error(error_message)
                    st.session_state.rag_chain = None
                else:
                    # Create vector store
                    vector_store = get_vector_store(transcript, gemini_api_key)
                    if vector_store:
                        # Create and store the RAG chain in session state
                        st.session_state.rag_chain = create_rag_chain(vector_store, gemini_api_key)
                        st.success("Video processed successfully! You can now ask questions.")
                        # Clear previous chat history when a new video is processed
                        st.session_state.messages = []
                    else:
                        st.error("Could not create vector store from the transcript.")
                        st.session_state.rag_chain = None

# --- CHAT INTERFACE ---

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask a question about the video..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if the RAG chain is ready
    if "rag_chain" not in st.session_state or st.session_state.rag_chain is None:
        st.warning("Please process a video first using the sidebar.")
    else:
        # Generate and display AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.invoke({"query": prompt})
                st.markdown(response)
        
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
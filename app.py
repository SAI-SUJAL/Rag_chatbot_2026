import streamlit as st
import os
import shutil
from huggingface_hub import InferenceClient
try:
    from huggingface_hub.utils._errors import HfHubHTTPError
except Exception:  # fallback for older/newer huggingface_hub versions
    from huggingface_hub.utils import HfHubHTTPError
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, PromptTemplate
from llama_index.core import StorageContext
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
try:
    from chromadb.errors import InvalidCollectionException
except Exception:  # fallback for older/newer Chroma versions
    InvalidCollectionException = Exception



# Use the maintained Chroma integration package
from langchain_chroma import Chroma

import chromadb
class _SimpleChatMemory:
    def __init__(self):
        self.messages = []

    def add_user_message(self, message):
        self.messages.append(f"User: {message}")

    def add_ai_message(self, message):
        self.messages.append(f"Assistant: {message}")


class ConversationBufferMemory:
    def __init__(self, memory_key="chat_history", return_messages=True):
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.chat_memory = _SimpleChatMemory()

    def load_memory_variables(self, inputs):
        # Returns a single key mapping with the conversation history
        history_str = "\n".join(self.chat_memory.messages)
        return {self.memory_key: history_str}
# Navigation links will use st.page_link; no extras needed

# Set page config
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# Hugging Face auth and model selection
HF_TOKEN = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
HF_MODEL = (
    st.secrets.get("HF_MODEL")
    or os.getenv("HF_MODEL")
    or "mistralai/Mistral-7B-Instruct-v0.2"
)
HF_MAX_NEW_TOKENS = int(
    st.secrets.get("HF_MAX_NEW_TOKENS") or os.getenv("HF_MAX_NEW_TOKENS") or 300
)

# Initialize your models, databases, and other components here
@st.cache_resource
def init_chroma():
    persist_directory = "chroma_db"
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    chroma_collection = chroma_client.get_or_create_collection("my_collection")
    return chroma_client, chroma_collection

@st.cache_resource
def init_vectorstore():
    persist_directory = "chroma_db"
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="my_collection",
    )
    return vectorstore

# Function to clear ChromaDB
def reset_chroma_db():
    global chroma_client, chroma_collection, vectorstore
    try:
        chroma_client.delete_collection("my_collection")
    except Exception:
        pass
    shutil.rmtree("chroma_db", ignore_errors=True)
    init_chroma.clear()
    init_vectorstore.clear()
    chroma_client, chroma_collection = init_chroma()
    vectorstore = init_vectorstore()
    st.success("ChromaDB reset successfully.")

# Guard against missing token
if not HF_TOKEN:
    st.error("Missing HF_TOKEN. Set it in .streamlit/secrets.toml or environment.")

# Initialize components
client = InferenceClient(model=HF_MODEL, token=HF_TOKEN)
chroma_client, chroma_collection = init_chroma()
vectorstore = init_vectorstore()

# Initialize memory buffer
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def _build_prompt(messages):
    user_content = "\n\n".join(
        [m["content"] for m in messages if m.get("role") == "user"]
    ) or messages[-1]["content"]
    return f"You are a helpful assistant.\n\n{user_content}\n\nAnswer:"


def _call_llm(messages):
    """Use chat_completion if available; fall back to text_generation."""
    try:
        result = client.chat_completion(
            messages=messages, max_tokens=HF_MAX_NEW_TOKENS, stream=False
        )
        return result.choices[0].message.content
    except HfHubHTTPError as e:
        if e.response is not None and e.response.status_code in (400, 401, 404, 422):
            # Fallback to text_generation endpoint
            prompt = _build_prompt(messages)
            return client.text_generation(
                prompt, max_new_tokens=HF_MAX_NEW_TOKENS, temperature=0.7
            )
        raise
    except (TypeError, AttributeError):
        # Older client without chat API
        prompt = _build_prompt(messages)
        return client.text_generation(
            prompt, max_new_tokens=HF_MAX_NEW_TOKENS, temperature=0.7
        )


def rag_query(query):
    try:
        # Retrieve relevant documents using similarity search from ChromaDB
        retrieved_docs = vectorstore.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else ""
    except Exception as e:
        # If the collection doesn't exist or there's any issue, skip to querying the LLM
        err_msg = str(e).lower()
        if "does not exist" in err_msg or "collection" in err_msg:
            try:
                reset_chroma_db()
            except Exception:
                pass
        context = ""

    # Append new interaction to memory
    memory.chat_memory.add_user_message(query)

    # Retrieve past interactions for context
    past_interactions = memory.load_memory_variables({})[memory.memory_key]
    context_with_memory = f"{context}\n\nConversation History:\n{past_interactions}"

    # Prepare message for LLM query
    messages = [
        {
            "role": "user",
            "content": f"Context: {context_with_memory}\n\nQuestion: {query}, it is not mandatory to use the context\n\nAnswer:"
        }
    ]

    # Get the response from the LLM (even if context is empty)
    try:
        llm_output = _call_llm(messages)
        response = llm_output.split("Answer:")[-1].strip()
    except Exception:
        st.error("LLM call failed. Check your HF token and network.")
        raise

    # If ChromaDB returns no context, rely solely on the LLM's response
    if not context or len(response.split()) < 35:
        # LLM handles the query without ChromaDB context
        messages = [{"role": "user", "content": query}]
        try:
            response = _call_llm(messages)
        except Exception:
            st.error("LLM call failed. Check your HF token and network.")
            raise

    # Append the response to memory
    memory.chat_memory.add_ai_message(response)

    return response


def process_feedback(query, response, feedback):
    if feedback:
        # If thumbs up, store the response in memory buffer
        memory.chat_memory.add_ai_message(response)
    else:
        # If thumbs down, regenerate the response
        new_query = f"{query}. Give a better response"
        new_response = rag_query(new_query)
        st.markdown(new_response)
        memory.chat_memory.add_ai_message(new_response)

# Streamlit interface
st.title("Welcome to our RAG-Based Chatbot")
st.markdown("*")
st.info('''
        To use our Mistral-supported Chatbot, click Chat.
         
        To push data, click on Store Document.
        ''')

st.subheader("Navigation")
st.page_link("pages/chatbot.py", label="Chat", icon="💬")
st.page_link("pages/management.py", label="Store Document", icon="📄")

# Button to clear ChromaDB
clear_data = st.button("Reset ChromaDB")
if clear_data:
    reset_chroma_db()

st.markdown("<div style='text-align:center;'></div>", unsafe_allow_html=True)

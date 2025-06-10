import gradio as gr
import os
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Configuration ---
# Path where your documents (PDFs, TXT) are located inside the Space
DOCS_DIR = "docs"
# Path where the FAISS index will be saved/loaded
FAISS_INDEX_PATH = "faiss_index"
# Hugging Face LLM model to use (smaller models are faster on CPU)
# Options: "google/flan-t5-base" (better quality, slower) or "google/flan-t5-small" (faster, less accurate)
LLM_MODEL_ID = "google/flan-t5-base" 
# Embedding model for converting text to vectors (small and efficient)
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# --- RAG Setup Functions ---
def load_documents(docs_dir):
    """Loads all PDF and TXT documents from the specified directory."""
    documents = []
    for root, _, files in os.walk(docs_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                if file_name.lower().endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                    print(f"Loaded {len(loader.load())} pages from {file_name}")
                elif file_name.lower().endswith('.txt'):
                    loader = TextLoader(file_path)
                    documents.extend(loader.load())
                    print(f"Loaded content from {file_name}")
                else:
                    print(f"Skipping unsupported file type: {file_name}")
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
    return documents

def setup_rag_pipeline():
    """Sets up the RAG pipeline components: splitter, embeddings, vector store, LLM, and QA chain."""

    print("Setting up RAG pipeline...")

    # 1. Load Documents
    print(f"Loading documents from {DOCS_DIR}...")
    documents = load_documents(DOCS_DIR)
    if not documents:
        raise ValueError(f"No documents found in {DOCS_DIR}. Please ensure your PDFs/TXTs are uploaded to this folder.")
    print(f"Total documents loaded: {len(documents)} pages/sections.")

    # 2. Text Splitting
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Documents split into {len(chunks)} chunks.")

    # 3. Embeddings (CPU-only)
    print(f"Loading Hugging Face Embeddings model: {EMBEDDING_MODEL_ID}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_ID,
        model_kwargs={'device': 'cpu'} # Ensure embeddings run on CPU
    )
    print("Embeddings model loaded.")

    # 4. Vector Store (FAISS) - Check if index exists, otherwise create
    # This allows the app to load faster on subsequent restarts after the first build.
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print("FAISS index loaded.")
    else:
        print("Creating FAISS vector store from chunks (first run, this might take a moment)...")
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        print("FAISS vector store created and saved.")

    # 5. Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    print("Retriever created.")

    # 6. LLM (CPU-only)
    print(f"Loading Hugging Face LLM: {LLM_MODEL_ID} (this will download model weights)...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_ID)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        device=-1 # -1 means CPU. Explicitly set for Spaces.
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    print(f"Hugging Face LLM '{LLM_MODEL_ID}' loaded.")

    # 7. RAG Chain
    prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer as concise as possible.

{context}

Question: {question}
Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)

    print("Creating the RAG chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    print("RAG chain created.")
    return qa_chain

# --- Global RAG Chain Initialization ---
# This will run once when the Gradio app starts
qa_chain_global = None
try:
    qa_chain_global = setup_rag_pipeline()
except Exception as e:
    print(f"Error during RAG pipeline setup: {e}")
    # You might want to display this error in the Gradio UI if setup fails completely.

# --- Gradio Interface ---

def chat_with_assistant(message, history):
    """Function to handle user queries and return AI responses with sources."""
    if qa_chain_global is None:
        return "The IT Assistant is not ready. There was an error during setup."

    print(f"\nUser query: {message}")
    start_query_time = time.time()

    # Invoke the RAG chain
    result = qa_chain_global.invoke({"query": message})

    end_query_time = time.time()

    response_text = result['result']
    source_documents = result['source_documents']

    # Format the response with sources
    formatted_response = response_text
    if source_documents:
        formatted_response += "\n\n**Sources:**\n"
        for i, doc in enumerate(source_documents):
            source_name = doc.metadata.get('source', 'Unknown Source')
            page_info = doc.metadata.get('page', 'N/A')
            # Include page content snippet in sources for debugging/demonstration
            # formatted_response += f"- {source_name} (Page: {page_info}): {doc.page_content[:100]}...\n"
            formatted_response += f"- {source_name} (Page: {page_info})\n"

    print(f"Assistant response: {response_text[:100]}...")
    print(f"Query time: {round(end_query_time - start_query_time, 2)} seconds")

    return formatted_response

# Create the Gradio interface
print("Launching Gradio interface...")
gr.ChatInterface(
    chat_with_assistant,
    chatbot=gr.Chatbot(height=500), # Adjust height for better display
    title="Internal IT Assistant (RAG Demo)",
    description="Ask me anything about our IT policies and FAQs. I'll retrieve answers from our internal knowledge base.",
    examples=[
        "How do I reset my password?",
        "What is the BYOD policy?",
        "Where is the data backup policy located?"
    ],
    theme="soft" # A nice, clean theme for Gradio
).queue().launch()
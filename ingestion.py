from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()  # Load environment variables (OpenAI API key)

urls = [
   "https://lilianweng.github.io/posts/2023-06-23-agent/",
   "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
   "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs =  [WebBaseLoader(url).load() for url in urls]  # Load content from each URL
docs_list = [item for sublist in docs for item in sublist]  # Flatten nested list of documents

# Create splitter with token-based counting
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
# Split documents into 250-token chunks
doc_splits = text_splitter.split_documents(docs_list)

# Create vector database from document chunks
vectorstore = Chroma.from_documents(
   documents=doc_splits,
   collection_name="rag_chroma",
   embedding=OpenAIEmbeddings(),  # Use OpenAI embeddings to convert text to vectors
   persist_directory="./chroma")  # Save database locally

# Create retriever interface for querying the vector database
retriever = Chroma(
   collection_name="rag_chroma",
   embedding_function=OpenAIEmbeddings(),
   persist_directory="./chroma"
).as_retriever()  # Convert to retriever object for RAG queries
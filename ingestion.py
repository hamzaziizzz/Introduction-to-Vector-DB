import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


if __name__ == "__main__":
    print("Ingestion started")

    text_loader = TextLoader("Hamza_Aziz_AI_Engineer_Resume.txt")
    pdf_loader = PyPDFLoader("Hamza_Aziz_AI_Engineer_Resume.pdf")

    document = text_loader.load()

    print("Splitting...")

    text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=16)
    texts = text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_TYPE"), model="text-embedding-3-small")

    print("Ingesting...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.getenv("PINECONE_INDEX"))

    print("Ingestion finished")

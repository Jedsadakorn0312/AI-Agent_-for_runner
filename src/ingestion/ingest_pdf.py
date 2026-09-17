from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def ingest_coach_data(
    pdf_path="data/data_runner.pdf",
    persist_directory="vector_store/coach_db"
):
    print(f"Loading PDF from {pdf_path}...")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print(f"Prepared {len(splits)} chunks from PDF.")
    print("Creating and persisting Vector Store for Coach Data...")

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    print("Coach data ingestion complete!")
    return vectorstore


if __name__ == "__main__":
    ingest_coach_data()
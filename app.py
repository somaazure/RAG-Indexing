import os
import hashlib
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.indexes import SQLRecordManager, index


class RAGIndexer:
    def __init__(self, document_path: str):
        self.document_path = document_path
        self.embedding_function = None
        self.vectorstore = None
        self.record_manager = None

    def load_environment(self):
        print("🔐 Loading environment variables...")
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY not found in .env file.")
        return api_key

    def compute_file_hash(self, filepath: str) -> str:
        print("🔍 Computing file hash for deduplication...")
        with open(filepath, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        print(f"✅ File hash: {file_hash}")
        return file_hash

    def load_documents(self, file_hash: str):
        print(f"📄 Loading documents from '{self.document_path}'...")
        loader = TextLoader(self.document_path)
        documents = loader.load()
        for doc in documents:
            doc.metadata["source"] = f"{self.document_path}_{file_hash}"
        print(f"✅ Loaded and tagged {len(documents)} document(s).")
        return documents

    def setup_embeddings(self, api_key):
        print("🧠 Setting up OpenAI Embeddings...")
        self.embedding_function = OpenAIEmbeddings(openai_api_key=api_key)

    def setup_vectorstore(self):
        print("🗃️ Setting up Chroma vector store...")
        self.vectorstore = Chroma(
            collection_name="rag_local",
            embedding_function=self.embedding_function,
            persist_directory="./chroma_store"
        )
        print("✅ Chroma store initialized at './chroma_store'.")

    def setup_record_manager(self):
        print("🧾 Initializing SQLRecordManager with SQLite for deduplication...")
        self.record_manager = SQLRecordManager(
            namespace="my_local_namespace",
            db_url="sqlite:///index_log.db"
        )
        self.record_manager.create_schema()
        print("✅ Record tracking database ready at 'index_log.db'.")

    def run_indexing(self):
        file_hash = self.compute_file_hash(self.document_path)
        documents = self.load_documents(file_hash)
        print("🚀 Starting indexing process with deduplication...")
        index(
            docs_source=documents,
            record_manager=self.record_manager,
            vector_store=self.vectorstore,
            cleanup="full",
            source_id_key="source"
        )
        print("🎉 Indexing complete!")

    def main(self):
        api_key = self.load_environment()
        self.setup_embeddings(api_key)
        self.setup_vectorstore()
        self.setup_record_manager()
        self.run_indexing()


# Entry point
if __name__ == "__main__":
    indexer = RAGIndexer(document_path="sample.txt")
    indexer.main()

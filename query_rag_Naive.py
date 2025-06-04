import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

"""“Naive RAG Implementation i.e typically retrieves documents using vector similarity and returns the raw content directly, 
without applying an LLM to interpret, summarize, or generate an answer."""

class RAGQueryTool:
    def __init__(self):
        self.load_environment()
        self.embedding_function = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.vectorstore = Chroma(
            collection_name="rag_local",
            embedding_function=self.embedding_function,
            persist_directory="./chroma_store"
        )

    def load_environment(self):
        print("🔐 Loading environment variables...")
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("❌ OPENAI_API_KEY not found in .env file.")
        print("✅ API Key loaded.")

    def query(self, user_question: str, k: int = 3):
        print(f"🔎 Querying for: {user_question}")
        results = self.vectorstore.similarity_search(user_question, k=k)
        print(f"\n🧠 Top {k} matching chunks:\n" + "-" * 40)
        for i, doc in enumerate(results, 1):
            print(f"[{i}] Source: {doc.metadata.get('source')}")
            print(doc.page_content.strip())
            print("-" * 40)


def main():
    tool = RAGQueryTool()
    print("💬 Ask a question or type 'exit' to quit.")
    while True:
        user_input = input("\n📝 Your question: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Exiting. Goodbye!")
            break
        if user_input:
            tool.query(user_input)


if __name__ == "__main__":
    main()

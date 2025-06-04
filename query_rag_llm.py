import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


class RAGQueryTool:
    def __init__(self):
        self.load_environment()
        self.embedding_function = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.vectorstore = Chroma(
            collection_name="rag_local",
            embedding_function=self.embedding_function,
            persist_directory="./chroma_store"
        )
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=self.api_key)

    def load_environment(self):
        print("🔐 Loading environment variables...")
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("❌ OPENAI_API_KEY not found in .env file.")
        print("✅ API Key loaded.")

    def generate_answer(self, context: str, question: str) -> str:
        messages = [
            SystemMessage(content="You are an expert assistant answering based only on the given context."),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}\nAnswer in one line:")
        ]
        response = self.llm(messages)
        return response.content.strip()

    def query(self, user_question: str, k: int = 3):
        print(f"\n🔎 Querying for: {user_question}")
        results = self.vectorstore.similarity_search(user_question, k=k)

        if not results:
            print("⚠️ No relevant context found.")
            return

        # Combine all retrieved content
        combined_context = "\n".join([doc.page_content for doc in results])
        answer = self.generate_answer(combined_context, user_question)

        print("\n💡 Answer:")
        print(answer)


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

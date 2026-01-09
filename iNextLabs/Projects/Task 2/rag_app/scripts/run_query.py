from app.retrieval import Retriever
from app.llm import call_llm


def main():
    query = input("Enter your question: ")

    retriever = Retriever(top_k=5)
    results = retriever.search(query)

    contexts = [chunk for chunk, _ in results]

    answer = call_llm(query, contexts)
    print("\n--- LLM Response ---\n")
    print(answer)


if __name__ == "__main__":
    main()

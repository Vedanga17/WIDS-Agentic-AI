
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
    You are an expert in answering questions about a pizza restaurant.
    Here are some reviews: {reviews}
    Answer the following question based on the reviews: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    question = input("Ask a question about the pizza restaurant (q to quit): ")
    if question.lower() == 'q':
        break
    reviews = retriever.invoke(question)

    result = chain.invoke({"reviews": reviews, "question": question})
    print("\n" + result)

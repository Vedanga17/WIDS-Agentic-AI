
from langchain_ollama import OllamaLLM # importing the LLM
from langchain_core.prompts import ChatPromptTemplate # importing prompt template
from vector import retriever # importing the retriever we created

model = OllamaLLM(model="llama3.2")

template = """
    You are an expert in answering questions about a pizza restaurant.
    Here are some reviews: {reviews}
    Answer the following question based on the reviews: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model # forming the chain

while True:
    question = input("Ask a question about the pizza restaurant (q to quit): ")
    if question.lower() == 'q':
        break
    reviews = retriever.invoke(question) # passing the question to the retriever to get relevant reviews for the LLM to use

    result = chain.invoke({"reviews": reviews, "question": question}) # passing the reviews retrieved and question to the LLM
    print("\n" + result)

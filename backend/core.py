import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2", model_kwargs={"device": "cpu"})

#initiate vector store
vector_store = PineconeVectorStore(
    index_name="doc-helper-index",
    embedding=embeddings,
)
retriever = vector_store.as_retriever(
    search_type="similarity", search_kwargs={"k": 5})

llm = ChatGoogleGenerativeAI(
    google_api_key=os.getenv("GEMINI_API_KEY"),
    model="gemini-2.5-flash",
    temperature=0.2,
)

template = """
You are a helpful assistant for the LangChain documentation.
Answer the question based *only* on the following context.
If you don't know the answer, just say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
#create chat prompt that can be poplulated with context and question
prompt = ChatPromptTemplate.from_template(template)

#format context into string
def format_docs(docs):
    formatted = ""
    for i, doc in enumerate(docs):
        formatted += f"--- Document {i+1} (Source: {doc.metadata.get('source', 'unknown')}) ---\n"
        formatted += doc.page_content + "\n\n"
    return formatted

#retrieval chain with LCEL this stuff be confusing asf
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
     | prompt
     | llm
     | StrOutputParser()
     
)

if __name__ == "__main__":
    question = "What is the simplest way to get started with LangChain?"
    
    print(f"Querying RAG chain with: '{question}'")
    answer = rag_chain.invoke(question)
    
    print("\n--- Answer ---")
    print(answer)

    # Example 2
    question_2 = "What is langchain?"
    print(f"\nQuerying RAG chain with: '{question_2}'")
    answer_2 = rag_chain.invoke(question_2)
    
    print("\n--- Answer ---")
    print(answer_2)
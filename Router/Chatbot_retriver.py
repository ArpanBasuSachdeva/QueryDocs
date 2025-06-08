from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from Router.exception_utils import log_exception
from Router.embedding import store_embeddings_in_chroma


load_dotenv()

# Initialize the retriever with similarity search
retriever = store_embeddings_in_chroma.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Initialize the language model with GPT-4
llm = ChatOpenAI(model="gpt-4", temperature=0.2)

# Create a prompt template for the chatbot
prompt = PromptTemplate(
    template="""
    You are a helpful AI assistant that provides accurate and relevant information.
    Use the following context to answer the user's question.
    If the context doesn't contain enough information, say you don't know.
    
    Context: {context}
    
    User Question: {question}
    
    Please provide a clear and concise answer:
    """,
    input_variables=['context', 'question']
)

def format_docs(retrieved_docs):
    """Format the retrieved documents into a single context string"""
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

# Create a parallel chain for processing
parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

# Create the main chain with output parsing
parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm | parser

def augmented_retrieval(user_question):
    """
    Function to handle user questions with retrieval augmentation
    
    Args:
        user_question (str): The question asked by the user
        
    Returns:
        str: The AI's response based on retrieved context
    """
    try:
        response = main_chain.invoke(user_question)
        return response
    except Exception as e:
        # Log the exception with details
        log_exception(e, "Error in chat_with_retrieval", {
            "user_question": user_question,
            "function": "chat_with_retrieval"
        })
        return f"Sorry, I encountered an error: {str(e)}"

# Example usage
if __name__ == "__main__":
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
        response = chat_with_retrieval(user_input)
        print(f"\nAssistant: {response}")
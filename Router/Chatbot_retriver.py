from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from Router.exception_utils import log_exception
from Router.embedding import DocumentLoaderManager


load_dotenv()

# Initialize the retriever with similarity search
try:
    vector_store = DocumentLoaderManager.load_embeddings()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
except Exception as e:
    print(f"Warning: Could not load embeddings: {e}")
    retriever = None

# Initialize the language model with GPT-4
try:
    llm = ChatOpenAI(model="gpt-4", temperature=0.2)
except Exception as e:
    print(f"Warning: Could not initialize OpenAI client: {e}")
    llm = None

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
    input_variables=['context', 'question'],   
)

def format_docs(retrieved_docs):
    """Format the retrieved documents into a single context string"""
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

# Create a parallel chain for processing
if retriever and llm:
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })
    
    # Create the main chain with output parsing
    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser
else:
    main_chain = None

def augmented_retrieval(user_question):
    """
    Function to handle user questions with retrieval augmentation
    
    Args:
        user_question (str): The question asked by the user
        
    Returns:
        str: The AI's response based on retrieved context
    """
    try:
        if retriever is None:
            return "I'm sorry, but I don't have access to any documents right now. Please upload a document first to enable context-aware responses."
        
        if llm is None:
            return "I'm sorry, but the AI language model is not available. Please set your OPENAI_API_KEY environment variable."
        
        if main_chain is None:
            return "I'm sorry, but the chat system is not properly initialized. Please check your configuration."
        
        # Add debug information to see if retrieval is working
        print(f"🔍 Searching for context related to: {user_question}")
        
        # Test retrieval separately for debugging
        retrieved_docs = retriever.invoke(user_question)
        print(f"📄 Found {len(retrieved_docs)} relevant documents")
        
        # Show first document snippet for debugging
        if retrieved_docs:
            print(f"📝 First document snippet: {retrieved_docs[0].page_content[:100]}...")
        
        response = main_chain.invoke(user_question)
        return response
    except Exception as e:
        # Log the exception with details - create own database session
        try:
            from Router.table_creater import SessionLocal
            db = SessionLocal()
            log_exception(e, "augmented_retrieval", {
                "user_question": user_question,
                "function": "augmented_retrieval"
            }, db=db)
            db.close()
        except Exception as log_error:
            # If database logging fails, just print
            print(f"Failed to log exception to database: {log_error}")
            log_exception(e, "augmented_retrieval", {
                "user_question": user_question,
                "function": "augmented_retrieval"
            })
        return f"Sorry, I encountered an error: {str(e)}"

# Example usage
if __name__ == "__main__":
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
        response = augmented_retrieval(user_input)
        print(f"\nAssistant: {response}")
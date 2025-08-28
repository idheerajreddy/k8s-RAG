from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    raise ValueError("Please set HUGGINGFACEHUB_API_TOKEN in your .env file")

# Load FAISS vector store
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
faiss_db = FAISS.load_local("faiss_index", embedder, allow_dangerous_deserialization=True)

# Set up retriever
retriever = faiss_db.as_retriever(search_kwargs={"k": 3})

# Set up LLM - Use HuggingFaceEndpoint directly (not ChatHuggingFace)
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",  # Use flan-t5-large instead of xl
    task="text2text-generation",
    temperature=0.1,
    max_new_tokens=512,
    huggingfacehub_api_token=hf_token
)
model = ChatHuggingFace(llm=llm)
# Create prompt template
prompt = PromptTemplate(
    template="""Answer the following question based only on the provided context. Be specific and helpful.

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# Create the chain using your RunnableMap approach
from langchain_core.runnables import RunnableMap

# First step: get docs and question
step1 = RunnableMap({
    "docs": RunnablePassthrough() | retriever,
    "question": RunnablePassthrough()
})

# Second step: format context and create prompt
def create_prompt_from_inputs(inputs):
    context = format_docs(inputs["docs"])
    question = inputs["question"]
    return f"Answer the following question using the provided context.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

step2 = step1.assign(
    context=lambda inputs: format_docs(inputs["docs"])
).assign(
    prompt=create_prompt_from_inputs
)

# Final step: get result
chain = step2.assign(
    result=lambda inputs: model.invoke(inputs["prompt"])
)

# Test the chain
if __name__ == "__main__":
    question = "How do I debug a Kubernetes cluster?"
    
    try:
        # Get the full output
        output = chain.invoke("How do I debug a Kubernetes cluster?")
        print("Question:", question)
        print("Answer:", output["result"])
        print("\nSource documents:")
        for i, doc in enumerate(output["docs"]):
            source = doc.metadata.get("source", "Unknown")
            print(f"{i+1}. {source}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

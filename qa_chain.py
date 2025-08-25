from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load FAISS vector store
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
faiss_db = FAISS.load_local("faiss_index", embedder, allow_dangerous_deserialization=True)

# Set up retriever
retriever = faiss_db.as_retriever()

# Set up LLM using HuggingFace Inference API (free)
llm_endpoint = HuggingFaceEndpoint(
    repo_id="google/flan-t5-xl",
    task="text2text-generation",
    huggingfacehub_api_token=hf_token
)
model = ChatHuggingFace(llm=llm_endpoint)

def format_prompt(question, docs):
    context = "\n\n".join([doc.page_content for doc in docs])
    return f"Answer the following question using the provided context.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

chain = (
    RunnableMap({
        "docs": RunnablePassthrough() | retriever,
        "question": RunnablePassthrough()
    })
    .assign(
        prompt=lambda inputs: format_prompt(inputs["question"], inputs["docs"])
    )
    .assign(
        result=lambda inputs: model.invoke(inputs["prompt"])
    )
)

# Ask a question
question = "How do I debug a Kubernetes cluster?"
output = chain.invoke(question)

print("Answer:", output["result"])
print("Source documents:", [doc.metadata["source"] for doc in output["docs"]])

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-Coder-7B-Instruct",
    task = "text-generation",
)

model = ChatHuggingFace(llm = llm)

result = model.invoke("What is capital of Germany?")
print(result.content)
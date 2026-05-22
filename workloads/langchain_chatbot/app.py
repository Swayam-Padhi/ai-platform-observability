from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langfuse.langchain import CallbackHandler
import time

# Load env
load_dotenv()

# Langfuse callback
langfuse_handler = CallbackHandler()

# Ollama model
llm = OllamaLLM(model="phi3")

# Prompts
prompts = [
    "Explain Kubernetes in simple words",
    "Explain Datadog in simple words with an example",
    "Explain Docker in simple words"
]

for prompt in prompts:

    response = llm.invoke(
        prompt,
        config={
            "callbacks": [langfuse_handler]
        }
    )

    print("\n====================")
    print(f"PROMPT: {prompt}")
    print("====================\n")

    print(response)

    # Give exporter time
    time.sleep(5)
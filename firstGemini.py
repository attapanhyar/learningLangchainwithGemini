import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Configure the Google API key
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# List available models to verify model names
genai.configure(api_key=google_api_key)
print("Available models:")
for model in genai.list_models():
    if "generateContent" in model.supported_generation_methods:
        print(model.name)

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Use a supported model (e.g., gemini-1.5-flash)
    google_api_key=google_api_key,
    temperature=0.7
)

# Create a prompt template
prompt = PromptTemplate.from_template("Write a short story about {topic}.")

# Create a RunnableSequence (replacing deprecated LLMChain)
chain = prompt | llm

# Run the chain with error handling
try:
    result = chain.invoke({"topic": "a magical forest"})
    print("\nGenerated Story:")
    print(result.content)
except Exception as e:
    print(f"Error: {e}")
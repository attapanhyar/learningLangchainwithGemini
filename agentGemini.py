import os
import wikipedia
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain import hub

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Use a supported model
    google_api_key=google_api_key,
    temperature=0.7
)

# Define a custom tool (calculate the cube of a number)
def cube_number(number: str) -> str:
    """Calculates the cube of a given number."""
    try:
        num = float(number)
        return str(num ** 3)
    except ValueError:
        return "Error: Please provide a valid number."

# Define a custom Wikipedia tool
def wikipedia_search(query: str) -> str:
    """Searches Wikipedia for a summary of the given query."""
    try:
        # Limit to 500 characters for brevity
        summary = wikipedia.summary(query, sentences=2, auto_suggest=True)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Error: Multiple results found for '{query}'. Try a more specific query."
    except wikipedia.exceptions.PageError:
        return f"Error: No Wikipedia page found for '{query}'."
    except Exception as e:
        return f"Error: {e}"

# Define tools
tools = [
    Tool(
        name="CubeCalculator",
        func=cube_number,
        description="Calculates the cube of a given number."
    ),
    Tool(
        name="WikipediaSearch",
        func=wikipedia_search,
        description="Searches Wikipedia for a summary of a topic."
    )
]

# Pull the ReAct prompt template from LangChain Hub
prompt = hub.pull("hwchase17/react")

# Create the ReAct agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Create the agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # Show reasoning steps
    handle_parsing_errors=True  # Handle Gemini's tool-calling quirks
)

# Run the agent with a query
try:
    result = agent_executor.invoke({
        "input": "What is the cube of 4? Also, tell me about the Amazon rainforest."
    })
    print("\nAgent Response:")
    print(result["output"])
except Exception as e:
    print(f"Error: {e}")
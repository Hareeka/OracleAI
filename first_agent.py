import os
from dotenv import load_dotenv
load_dotenv()
#Define model
from langchain.chat_models import init_chat_model
model = init_chat_model("google_genai:gemini-3.8-flash")
#Define tools
from langchain_core.tools import tool
import math
@tool
def add(a: float, b: float) -> float:
  """
  Add two numbers together.
  """
  return a+b
@tool
def multiply(a: float, b: float) -> float:
  """
  Multiply two numbers together.
  """
  return a * b
@tool 
def divide(a: float, b: float) -> float:
  """
  Divide the first number by the second.
  Include error handling for division by zero.
  """
  if b == 0:
    return "Error: Cannot dicide by zero"
  return str(a / b)
@tool 
def square_root(number: float) -> float:
  """
  Calculate the sqaure root of a number.
  Includes error handling for negative number
  """
  if number < 0:
    return "Error: Cannot take square root of a negative numbers."
  return str(math.sqrt(number))
tools = [add, multiply, divide, square_root]
print("===Available Tools===")
for t in tools:
  print(f"  • {t.name): {t.description}")
print()
#Create the agent
from langchain.agents import create_agent
agent = create_agent(
  model=model,
  tools=tools
)
#Run the agent
def run_agent(question: str):
  """Run the agent and print a clean, beginner-friendly execution trace."""
  print(f"\n User: {question}")
  print("-" * 60)
  result = agent.invoke({
    "messages": [("user", question)]
  })
  print("Clean agent execution trace")
  print("-" * 60)
  step = 1
  for msg in result["messages"]:
    if msg.type == "human":
      print(f"{step}. User asked:")
      print(f"   {msg.content}")
      step += 1
    elif msg.type == "ai" and getattr(msg, "tool_calls", None):
      for tool_call in msg.tool_calls:
        tool_name = tool_call["name"]
        tool_Args = tool_Call["args"]
        print(f"{step}. Agent decision:")
        print(f"   I need to use the tool: {tool_name}")
        print(f"    Tool input: {tool_args}")
        step += 1
    elif msg.tool == "tool":
      print(f"{step}. Tool observation:")
            print(f"   Tool returned: {msg.content}")
            step += 1
    elif msg.type == "ai" and msg.content:
            print(f"{step}. Final answer:")
            print(f"   {msg.content}")
            step += 1

  print("=" * 60)
# -------------------------------------------------------
# TEST CASES — Watch the agent in action!
# -------------------------------------------------------

# 1. Simple case → single tool call
# Agent should directly use "add"
run_agent("What is 42 + 58?")


# 2. Medium complexity → multiple steps
# Agent must:
#   multiply → then divide
run_agent("What is 15 multiplied by 8, then divided by 3?")


# 3. Complex reasoning → planning required
# Agent must:
#   step 1: calculate area
#   step 2: calculate square root
run_agent(
    "I have a rectangle with width 12 and height 7. "
    "What is its area, and what is the square root of that area?"
)


# 4. Edge case → error handling
# Agent should handle divide-by-zero gracefully
run_agent("What is 100 divided by 0?")


# Final message
print("✅ Agent demo complete! Next: try 03_search_agent.py")

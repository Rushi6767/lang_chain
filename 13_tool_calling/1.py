from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# tool create

@tool
def multiply(a: int, b: int) -> int:
  """Given 2 numbers a and b this tool returns their product"""
  return a * b

# print(multiply.invoke({'a':3, 'b':4}))

# print(multiply.name)
# print(multiply.description)

# print(multiply.args)

# tool binding
llm = ChatOpenAI()

# llm.invoke('hi')

llm_with_tools = llm.bind_tools([multiply])

# respose from llm
# print(llm_with_tools.invoke('Hi how are you'))

# response from llm but suggest not calling
# print(llm_with_tools.invoke('multiple 10 with 6'))

query = HumanMessage('can you multiply 3 with 1000')
messages = [query]

# print(messages)

result = llm_with_tools.invoke(messages)
messages.append(result)

tool_result = multiply.invoke(result.tool_calls[0])
# print(tool_result)

messages.append(tool_result)
print(messages)

final_result = llm_with_tools.invoke(messages).content
print(final_result)
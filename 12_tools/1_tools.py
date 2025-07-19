"""
Built_in tools Duckduckgo search
"""
from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

results = search_tool.invoke('top news in india today')

print(results)

print(search_tool.name)
print(search_tool.description)
print(search_tool.args)


"""
shell tools
"""
# from langchain_community.tools import ShellTool

# shell_tool = ShellTool()

# results = shell_tool.invoke('ls')

# print(results)
"""follow this this ia mentioned in latest code"""

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

# detailed way
template2 = PromptTemplate(
    template='five things about {name}',
    input_variables=['name']
)

# fill the values of the placeholders
prompt = template2.invoke({'name':'India'})

result = model.invoke(prompt)

print(result.content)

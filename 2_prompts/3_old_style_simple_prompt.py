from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables (make sure you have .env with OPENAI_API_KEY)
load_dotenv()

# Initialize OpenAI model
llm = ChatOpenAI()

# Define a simple prompt template
prompt = PromptTemplate.from_template(
    "Explain the concept of {topic} in a way that a {audience} can understand."
)

# Create a prompt with values
final_prompt = prompt.format(topic="machine learning", audience="beginner")

# Run the model
response = llm.invoke(final_prompt)

# Print result
print("🔍 Prompt:", final_prompt)
print("🧠 Response:", response.content)

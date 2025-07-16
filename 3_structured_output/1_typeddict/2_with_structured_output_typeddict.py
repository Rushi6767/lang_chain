from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model = ChatOpenAI()

# schema
class Review(TypedDict):

    summary: str
    sentiment: str   

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""Chaava is an absolute cinematic masterpiece that brings the legend of Maharaja Chhatrapati Sambhaji Maharaj 
                                 to life with breathtaking visuals and emotionally charged storytelling. Every scene is a visual wonder, immersing
                                  the audience in the grandeur of history while delivering powerful moments that send chills down the spine.
""")

print(type(result))
print(result)

print(result['summary'])
print(result['sentiment'])
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

video_id = "Gfr50f6ZBvo" # only the ID, not full URL
try:
    # If you don’t care which language, this returns the “best” one
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])

    # Flatten it to plain text
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    # print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")

# print(transcript_list)

# indexing
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

# print(len(chunks))
# print(chunks[100])

# Embedding Generation
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Storing in vectore store
vector_store = Chroma.from_documents(chunks, embeddings)

# View stored documents
stored_docs = vector_store.get(include=["documents", "metadatas"])
print(f"Total documents stored: {len(stored_docs['documents'])}")

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
print(retriever)

# i = retr


# augument
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs = retriever.invoke(question)

# print(retrieved_docs)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
# print(context_text)

final_prompt = prompt.invoke({"context": context_text, "question": question})

# print(final_prompt)

# generation
answer = llm.invoke(final_prompt)
print(answer.content)
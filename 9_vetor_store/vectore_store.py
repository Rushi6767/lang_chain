from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

from langchain.schema import Document

# Create LangChain documents for IPL players

doc1 = Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
    )
doc2 = Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
    )
doc3 = Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
    )
doc4 = Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
    )
doc5 = Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
    )

docs = [doc1, doc2, doc3, doc4, doc5]

vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory='my_chroma_db',
    collection_name='sample'
)

# run one time
"""
Every time you call add_documents(docs), the Chroma vector store adds new entries, even if their contents are identical to previously added documents. This results in duplication because:
add_documents() doesn't check for duplicates.
Unless you manually delete or prevent re-inserts, it keeps accumulating.
Your persist_directory='my_chroma_db' is storing this between runs unless you reset it.
"""
# vector_store.add_documents(docs)
# print(vector_store.add_documents(docs))

# solution
vector_store.delete(ids=vector_store.get()['ids'])  # delete all
vector_store.add_documents(docs)

# view documents
data = vector_store.get(include=['embeddings','documents', 'metadatas'])
print(data)

# search documents
search = vector_store.similarity_search(
    query='Who among these are a bowler?',
    k=2
)
print(search)

# search with similarity score
search_similarity_score = vector_store.similarity_search_with_score(
    query='Who among these are a bowler?',
    k=2
)
# print(search_similarity_score)

# meta-data filtering
sss = vector_store.similarity_search_with_score(
    query="",
    filter={"team": "Chennai Super Kings"}
)
# print(sss)


# update documents
updated_doc1 = Document(
    page_content="Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consistent batting performances. He holds the record for the most runs in IPL history, including multiple centuries in a single season. Despite RCB not winning an IPL title under his captaincy, Kohli's passion and fitness set a benchmark for the league. His ability to chase targets and anchor innings has made him one of the most dependable players in T20 cricket.",
    metadata={"team": "Royal Challengers Bangalore"}
)

vector_store.update_document(document_id='09a39dc6-3ba6-4ea7-927e-fdda591da5e4', document=updated_doc1)

upadted_data = vector_store.get(include=['embeddings','documents', 'metadatas'])

# print(upadted_data)

# delete 7aedd91d-d7ac-46bb-a7e6-62c5d8b20514

# delete document
vector_store.delete(ids=['09a39dc6-3ba6-4ea7-927e-fdda591da5e4'])
data = vector_store.get(include=['embeddings','documents', 'metadatas'])
# print(data)

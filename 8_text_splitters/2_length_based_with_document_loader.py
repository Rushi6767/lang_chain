from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('Resume.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=200,
    # if 10 then 10 words comman in both
    chunk_overlap=0,
    separator=''
)

result = splitter.split_documents(docs)

print(result)
print(result[1].page_content)
# push data into vector DB
from src.helper import load_pdf, text_split,download_hugging_face_embeddings
import pinecone
from dotenv import load_dotenv 
from langchain_pinecone import PineconeVectorStore
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
index_name = os.environ.get('index_name')

print(PINECONE_API_KEY)
print(index_name)


#01. load the PDF
extracted_data = load_pdf("data/")

#02. Split into chunks
text_chunks = text_split(extracted_data)

#03 . Load the embedding model 
embeddings = download_hugging_face_embeddings()

#04. Initialize the PineCone
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
index_name = index_name

docsearch = PineconeVectorStore.from_texts(
        [t.page_content for t in text_chunks],
        index_name=index_name,
        embedding=embeddings
    )
from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from src.prompt import *
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import pinecone
from dotenv import load_dotenv
import os 

app= Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
index_name = os.environ.get('index_name')

#03 . Load the embedding model 
embeddings = download_hugging_face_embeddings()

#04. Initialize the PineCone
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
index_name = index_name

# 05 if you created index then dont use below mentioned code

# docsearch = PineconeVectorStore.from_texts(
#         [t.page_content for t in text_chunks],
#         index_name=index_name,
#         embedding=embeddings
#     )

# 06.
#If we already have an index we can load it like this
docsearch=PineconeVectorStore.from_existing_index(index_name, embeddings)

#07 Load Prompt Template
PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

# 08. Load the LLM Model
llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})

# 09. Initiate the QA Chain
qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
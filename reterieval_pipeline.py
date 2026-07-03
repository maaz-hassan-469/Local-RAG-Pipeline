from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

#here it is not necessary becuase we are using local huggingface model
load_dotenv()
#load the database from vector database db/chroma_db and use the same embedding model that we used for ingestion
persistent_directory="db/chroma_db"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#here we create db object using chroma class and pass the arguments 
db=Chroma(
persist_directory=persistent_directory,
embedding_function=embedding_model,
collection_metadata={"hnsw:space": "cosine"}
)

#taking input from the user for query related tp dcouments

query=input("Enter your query related to the documents you ingested:")

#search_kwargs means that we want to retrieve the top 5 most relevant documents from vector database
retriever=db.as_retriever(search_kwargs={"k":5})

#invoke is the method that we use to retrieve the relevant documents from the vector database based on the user query.
#  It takes the user query as an argument and returns a list of relevant documents.
# also it embedding the user query using the same model 
relevant_docs=retriever.invoke(query)

print(f"user query: {query}")
print("---context---")

for i,doc in enumerate(relevant_docs,1):
    print(f"Document {i}:\n {doc.page_content}\n")

# #creating input for the llm model to generate answer based on the user query and relevant documents retrieved from vector database 
# #in the second line we are joining the content of all the relevant documents into a single string using chr(10) which is the newline character.
# #using loop on relevant_docs to get the page_content of each document and join them with newline character.
# combined_input=f"""Based on the following documents, please answer this question: {query}\n
# # Documents: {chr(10).join([doc.page_content for doc in relevant_docs])}

# # please provide a clear,helpful answer using only the information from these documents.If the answer is not contained within the documents say "i dotn have enough information to answer the question" and do not make up an answer
# # """

# model=ChatOpenAI(model="gpt-4o")
# #here we are creating a list of message to pass the model
# #system message is used to set the behacior of the model
# #and human message is used to pass the user query and relevant documents to the model
# messages=[SystemMessage(content="you are helpful assistant"),
#           HumanMessage(content=combined_input)]

# #here we are invoking the model with the messages we created above and storing the result in a variable called result
# result=model.invoke(messages)


# print("\n ---generted answer--- ")

# print("content only:")
# print(result.content)





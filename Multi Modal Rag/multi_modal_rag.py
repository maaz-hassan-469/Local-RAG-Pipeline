import json
import os
from typing import List
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.staging.base import elements_to_json, elements_from_json
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_community.embeddings import FastEmbedEmbeddings

file_path = "D:/RAG/RAG/retrieval-augmented-generation-options.pdf"
CACHE_JSON_PATH = "partitioned_elements.json"
v_llm=ChatOllama(model="moondream", temperature=0.2)
qa_llm = ChatOllama(model="llama3.2", temperature=0.1)
embedding_model=FastEmbedEmbeddings()
CHROMA_DIR = "D:/RAG/RAG/chroma_db"


def partition_document(file_path:str):
    """Extract elements from PDF using unstructured"""
    print(f"partioning documents:{file_path}")

#check if elements are already exists in the cache
    if os.path.exists(CACHE_JSON_PATH):
        print("loading elements from cache")
        elements=elements_from_json(CACHE_JSON_PATH)
        print(f"Loaded {len(elements)} elements from disk!")
        return elements

#run if elements are not in cache
    elements=partition_pdf(
        filename=file_path,
        strategy="hi_res",#use the most accurate (but slower) processing method for extraction
        extract_image_block_types=["Image"],#Grab images found in PDF
        infer_table_structure=True,#keep table as structure HTML, not jumbled text
        extract_image_block_to_payload=True#store image as base64 data that ucan actually use
    )

    elements_to_json(elements, filename=CACHE_JSON_PATH)
    print(f"✅ Saved {len(elements)} elements to '{CACHE_JSON_PATH}' for future runs!")
    return elements

def create_chunks_by_title(elements):
    """create intelligent based chunking by using title based chunking"""

    print("creating smart chunks")
    chunks=chunk_by_title(
        elements,
        max_characters=3000,#hard limit never exceed 3000 characters per chunk
        new_after_n_chars=2400,#try to start new chunk after 2400
        combine_text_under_n_chars=500#merge tiny chunks under 500 chars with neighbours
    )

    print(f"create {len(chunks)} chunks")

    return chunks

def separate_content(chunks):
    content_data={
        "text":chunks.text,
        "images":[],
        "tables":[],
        "types":["text"]
    }

    if hasattr(chunks,"metadata") and hasattr(chunks.metadata,"orig_elements"):
        for element in chunks.metadata.orig_elements:
            element_type=   type(element).__name__

            if element_type=="Table":
                content_data["types"].append("table")
                table_html=getattr(element.metadata,"text_as_html",element.text)
                content_data["tables"].append(table_html)

            elif element_type=="Image":
                if hasattr(element,"metadata") and hasattr(element.metadata,"image_base64"):
                    content_data["types"].append("image")
                    content_data["images"].append(element.metadata.image_base64)

    content_data["types"]=list(set(content_data["types"]))
    return content_data

def summarize_Chunks(chunks):
    """process all chunks with AI summaries"""
    print("processing chunnks with ai memories")
    langchain_documents=[]
    total_chunks=len(chunks)

    for i,chunk in enumerate(chunks):
        current_chunk=i+1
        print(f"Processing chunk {current_chunk}/{total_chunks}...")

        content_data=separate_content(chunk)
        print(f"      Types found: {content_data['types']}")
        print(f"      Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}")

        if content_data['tables'] or content_data['images']:
            print(f"-> Creating AI summary for mixed content...")
            try:
                enhanced_content = create_ai_enhanced_summary(
                content_data['text'],
                content_data['tables'],
                content_data['images']
                        )
                print("ai summary created successfully")
            except Exception as e:
                print("ai summary failed")
                enhanced_content=content_data["text"]

        else:
            print("using raw text")
            enhanced_content=content_data["text"]

        doc=Document(
            page_content=enhanced_content,
            metadata={
                "original_content":json.dumps({
                    "raw_text":content_data["text"],
                    "tables_html":content_data["tables"],
                    "image_base64":content_data["images"]
                })
            }       
              )

        langchain_documents.append(doc)
        print(f"processed {len(langchain_documents)} chunks")

    return langchain_documents
            


def create_ai_enhanced_summary(text:str,tables:List[str],images:List[str])-> str:
    """create AI-enhanced summary for mixed content"""
    try:
       
        prompt_text=f"""you are creating a searchable description for document content retrieval.
        CONTENT TO ANALYZE:
        TEXT CONTENT:
        {text}"""

        if tables:
            prompt_text+="Tables:\n"
            for i,table in enumerate(tables):
                prompt_text+=f"Table {i+1}:\n {table}\n\n"

        prompt_text+="""YOUR TASK:
                Generate a comprehensive, searchable description that covers:

                1. Key facts, numbers, and data points from text and tables
                2. Main topics and concepts discussed
                3. Questions this content could answer
                4. Visual content analysis (charts, diagrams, patterns in images)
                5. Alternative search terms users might use

                Make it detailed and searchable - prioritize findability over brevity.

                SEARCHABLE DESCRIPTION:"""

        message_content = [{"type": "text", "text": prompt_text}]
        for image_base64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": {"url":f"data:image/jpeg;base64,{image_base64}"}
            })
        message = HumanMessage(content=message_content)
        response = v_llm.invoke([message])

        return response.content

    except Exception as e:
        print("ai summary failed")
        summary=f"{text[:300]}"
        if tables:
            summary+=f"contains {len(tables)} tables"
        if images:
            summary+=f"contains {len(images)} images"
        return summary
    
def export_chunks_to_json(chunks,filename="chunks_export.json"):
    """export processed chunk into json"""
    export_data=[]
    for i,doc in enumerate(chunks):
        chunk_data={
            "chunk_id":i+1,
            "enhanced_content":doc.page_content,
            "metadata":{
                "original_content":json.loads(doc.metadata.get("original_content","{}"))
            }
        }
        export_data.append(chunk_data)

    with open(filename,"w",encoding="utf-8") as f:
        json.dump(export_data,f,indent=2,ensure_ascii=False)
    print(f"exported {len(export_data)} chunks to {filename}")
    return export_data

def get_or_create_vectorstore(pdf_path: str, persist_directory: str =CHROMA_DIR):
    """Loads existing Chroma vectorstore or runs the full ingestion pipeline."""
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print(f"Loading existing vectorstore from '{persist_directory}'...")
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"},
        )

    print(f"No existing vectorstore found. Starting full ingestion...")
    elements = partition_document(pdf_path)
    chunks = create_chunks_by_title(elements)
    summarized_chunks = summarize_Chunks(chunks)

    print(f"\n[4/4] Creating Vectorstore in '{persist_directory}'...")
    vector_store = Chroma.from_documents(
        documents=summarized_chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},  # Fixed typo
    )
    print("Vectorstore initialized successfully.")
    return vector_store

def answer_query(query:str,vector_store:Chroma,k: int=3):
    """retrieve relevant chunks and generate an answer with llm"""
    retriever=vector_store.as_retriever(search_kwargs={"k":k})
    retrieved_docs=retriever.invoke(query)

    context_blocks = []
    for i, doc in enumerate(retrieved_docs):
        # Prefer original text + summary for rich grounding
        raw_text = doc.metadata.get("raw_text", "")
        summary = doc.page_content
        context_blocks.append(
            f"--- Context Block {i+1} ---\n"
            f"Summary/Search Index: {summary}\n"
            f"Original Content: {raw_text}\n"
        )

    context_str = "\n".join(context_blocks)

    system_prompt = (
        "You are an expert technical assistant. Answer the user's question using ONLY the provided context. "
        "If the answer cannot be deduced from the context, state that clearly. "
        "Be factual, well-structured, and concise."
    )
    user_prompt=f"CONTEXT:\n{context_str}\n\nUSER QUESTION: {query}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    print("\nGenerating response...")
    response = qa_llm.invoke(messages)
    return response.content, retrieved_docs

def main():
    db=get_or_create_vectorstore(file_path,persist_directory=CHROMA_DIR)
    print("\n" + "=" * 50)
    print("RAG System Ready! Type 'exit' or 'q' to quit.")
    print("=" * 50 + "\n")
    while True:
        try:
            user_input=input("enter your question: ").strip()
            if user_input.lower() in ["exit","q","quit"]:

                print("exiting...")
                break
            if not user_input:
                continue

            answer,docs=answer_query(user_input,db,k=3)
            print("\n" + "-" * 40)
            print("ANSWER:")
            print("-" * 40)
            print(answer)
            print("-" * 40)
            print(f"(Retrieved {len(docs)} supporting context chunks)\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
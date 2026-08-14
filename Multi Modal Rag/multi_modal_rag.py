import json
import os
from typing import List
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.staging.base import elements_to_json, elements_from_json
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_community.embeddings import FastEmbedEmbeddings

file_path = "D:/RAG/RAG/retrieval-augmented-generation-options.pdf"
CACHE_JSON_PATH = "partitioned_elements.json"

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

    for i,chunks in enumerate(chunks):
        current_chunk=i+1
        print("Processing chunk {current_chunk}/{total_chunks}...")

        content_data=separate_content(chunks)
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
        print("processed {len(langchain_documents)} chunks")

        return langchain_documents
            


def create_ai_enhanced_summary(text:str,tables:List[str],images:List[str])-> str:
    """create AI-enhanced summary for mixed content"""
    try:
        llm=ChatOllama(model="llama3.2-vision:11b", temperature=0.2)
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
        response = llm.invoke([message])

        return response.content

    except Exception as e:
        print("ai summary failed")
        summary=f"{text[:300]}"
        if tables:
            summary+=f"contains {len(tables)} tables"
        if images:
            summary+=f"contains {len(images)} images"
        return summary
    








Element=partition_document(file_path)
chunks=create_chunks_by_title(Element)
summaries=summarize_Chunks(chunks)

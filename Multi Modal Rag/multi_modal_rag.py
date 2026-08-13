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


Element=partition_document(file_path)
chunks=create_chunks_by_title(Element)

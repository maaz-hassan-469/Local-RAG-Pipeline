import json
from typing import List
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_community.embeddings import FastEmbedEmbeddings

def partition_document(file_path:str):
    """Extract elements from PDF using unstructured"""
    print(f"partioning documents:{file_path}")

    elements=partition_pdf(
        filename=file_path,
        strategy="hi_res",#use the most accurate (but slower) processing method for extraction
        extract_image_block_types=["Image"],#Grab images found in PDF
        infer_table_structure=True,#keep table as structure HTML, not jumbled text
        extract_image_block_to_payload=True#store image as base64 data that ucan actually use
    )

    print(f"extract {len(elements)} elements")
    return elements


import os
from langchain.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader, UnstructuredExcelLoader, UnstructuredPowerPointLoader)
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
import tempfile
import pinecone
import boto3
from botocore.exceptions import NoCredentialsError

# def load_documents(directory_path):
#     try:
#         documents = []
#         if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
#             raise ValueError(f"Invalid directory path: {directory_path}")

#         for file in os.listdir(directory_path):
#             file_path = os.path.join(directory_path, file)
#             if os.path.isfile(file_path):
#                 ext = os.path.splitext(file)[-1].lower()
#                 loaders = {
#                     ".pdf": PyPDFLoader,
#                     ".docx": Docx2txtLoader,
#                     ".doc": Docx2txtLoader,
#                     ".txt": TextLoader,
#                     ".csv": CSVLoader,
#                     ".xlsx": UnstructuredExcelLoader,
#                     ".xls": UnstructuredExcelLoader,
#                     ".pptx": UnstructuredPowerPointLoader
#                 }
#                 loader = loaders.get(ext)
#                 if loader:
#                     documents.extend(loader(file_path).load())
#     except Exception as e:
#         print(f"Error loading documents: {e}")
#         documents = []
#     return documents

def load_documents(bucket_name, s3_prefix):
    try:
        documents = []
        
        # Create a temporary directory to store files
        temp_dir = tempfile.mkdtemp()
        
        # Initialize the S3 client
        s3 = boto3.client('s3')
        
        # List objects in the S3 bucket
        objects = s3.list_objects(Bucket=bucket_name, Prefix=s3_prefix)['Contents']
        
        for obj in objects:
            key = obj['Key']
            _, file_extension = os.path.splitext(key)
            file_extension = file_extension.lower()
            
            # Map file extensions to loaders
            loaders = {
                ".pdf": PyPDFLoader,
                ".docx": Docx2txtLoader,
                ".doc": Docx2txtLoader,
                ".txt": TextLoader,
                ".csv": CSVLoader,
                ".xlsx": UnstructuredExcelLoader,
                ".xls": UnstructuredExcelLoader,
                ".pptx": UnstructuredPowerPointLoader
            }
            
            loader = loaders.get(file_extension)
            
            if loader:
                # Download the file from S3
                file_path = os.path.join(temp_dir, os.path.basename(key))
                s3.download_file(bucket_name, key, file_path)
                
                # Load the document using the specified loader
                documents.extend(loader(file_path).load())
                
        # Clean up temporary directory
        for file_name in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file_name)
            os.remove(file_path)
        os.rmdir(temp_dir)
        
    except NoCredentialsError:
        raise ValueError("AWS credentials not found. Please configure your AWS credentials.")
    except Exception as e:
        print(f"Error loading documents: {e}")
        documents = []
        
    return documents

def split_documents(documents, chunk_size=2000, chunk_overlap=20):
    try:
        if not documents:
            raise ValueError("No documents provided for splitting.")
        
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than 0.")
        
        if chunk_overlap < 0:
            raise ValueError("Chunk overlap must be non-negative.")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
        
        return text_splitter.split_documents(documents)
    
    except ValueError as ve:
        print(f"Error in split_documents: {ve}")
        return [] 

def user_model(documents=None, model_name='gpt-3.5-turbo'):
    try:
        if documents is None:
            raise ValueError("No documents provided for model creation.")
        
        if not isinstance(documents, list) or len(documents) == 0:
            raise ValueError("Invalid 'documents' parameter. It should be a non-empty list of documents.")
        pinecone.init()
        vectordb = Pinecone.from_documents(documents, embedding=OpenAIEmbeddings())
        memory = ConversationBufferMemory(memory_key="chat_history",
                                          return_messages=True,
                                          output_key='answer')

        llm = ChatOpenAI(temperature=0.7, model_name=model_name)
        model = ConversationalRetrievalChain.from_llm(llm,
                                                      retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
                                                      return_source_documents=True,
                                                      memory=memory,
                                                      verbose=False)

        return model

    except ValueError as ve:
        print(f"Error in user_model: {ve}")
        return None 


class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


def process_metadata(documents):
    source_to_pages = {}
    unique_sources = set()
    unique_pages = set()

    try:
        for document in documents:
            metadata = document.metadata
            page = metadata.get('page', 'NA')
            source = metadata.get('source', 'NA')

            unique_pages.add(page)
            unique_sources.add(source)

            if source not in source_to_pages:
                source_to_pages[source] = set()

            source_to_pages[source].add(page)

        if len(unique_sources) == 1 and len(unique_pages) > 1:
            source_value = next(iter(unique_sources))
            return f"Pages: {', '.join(map(str, unique_pages))}, Source: {source_value}"
        elif len(unique_sources) == 1 and len(unique_pages) == 1:
            page_value = next(iter(unique_pages))
            source_value = next(iter(unique_sources))
            return f"Page: {page_value}, Source: {source_value}"
        else:
            result = []
            for source, pages in source_to_pages.items():
                pages_str = ', '.join(map(str, pages))
                result.append(f"Source: {source}, Pages: {pages_str}")
            return result

    except Exception as e:
        return f"Error processing metadata: {str(e)}"





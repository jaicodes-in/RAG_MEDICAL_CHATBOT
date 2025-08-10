import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

from app.common.custom_exception import CustomException
from app.common.logger import get_logger

from app.config.config import DATA_PATH,CHUNK_OVERLAP,CHUNK_SIZE

logger=get_logger(__name__)
splitter_obj=RecursiveCharacterTextSplitter()


def load_pdf_files():

    try:
        if not os.path.exists(DATA_PATH):
            raise CustomException("Data path doesn't exist")
        logger.info(f'Loading files from {DATA_PATH}')

        loader=DirectoryLoader(DATA_PATH,glob="*.pdf",loader_cls=PyPDFLoader)

        documents= loader.load()

        if not documents:
            logger.warning(f'No PDFs found')
        else:
            logger.info(f'Successfully fetched {len(documents)}documents')

        return documents

    except Exception as e:
        error_message=CustomException('Failed to load PDF due ',error_detail=e)
        logger.error(str(error_message))
        return []


def create_text_chunks(documents:List):
    try:

        if not documents:
            raise CustomException('no documents were found')
        
        logger.info(f'Splitting  {len(documents)} documents into chunks')

        splitter = RecursiveCharacterTextSplitter()
        
        text_chunks = splitter.split_documents(documents=documents)

        logger.info(f'Generated {len(text_chunks)} chunks after chunking')
        
        return text_chunks
    
    except Exception as e:
        error_message = CustomException('Failed to generate chunks',error_detail=e)
        logger.error(str(error_message))
        return []
    




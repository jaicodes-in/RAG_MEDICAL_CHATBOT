import os

from app.components.pdf_loader import load_pdf_files,create_text_chunks

from app.components.vectorstore import save_vector_store,load_vector_store

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger=get_logger(__name__)


def process_and_store_pdfs():

    try:
        logger.info('making vector database.....')

        documents=load_pdf_files()

        text_chunks= create_text_chunks(documents=documents)

        save_vector_store(text_chunks=text_chunks)

    except Exception as e:
        error_message = CustomException('Failed to process PDFs and create vectorstore',error_detail=e)
        logger.error(str(error_message))

if __name__=="__main__":
    process_and_store_pdfs()
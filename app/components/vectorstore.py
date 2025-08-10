from langchain_community.vectorstores import FAISS
import os
from app.components.embeddings import get_embedding_model

from app.common.custom_exception import CustomException
from app.common.logger import get_logger

from app.config.config import DB_FAISS_PATH

logger=get_logger(__name__)

def load_vector_store():
    try:
        embedding_model=get_embedding_model()
        if os.path.exists(DB_FAISS_PATH):
            logger.info(f'Loading Existing Vector store')
            return FAISS.load_local(
                                        DB_FAISS_PATH,
                                        embeddings=embedding_model,
                                        allow_dangerous_deserialization=True
                                    )   

        else:
            logger.warning(f'No vector store detected or found....')
    
    except Exception as e:
        error_message = CustomException('Failed to load the vector store',error_detail=e)
        logger.error(str(error_message))

def save_vector_store(text_chunks): ## 
    """For creating a new vector"""
    try:
        if not text_chunks:
            raise CustomException('no chunks were found..')
        
        logger.info('Generating your new vector store')

        embedding_model=get_embedding_model()

        db=FAISS.from_documents(
            documents=text_chunks,
            embedding=embedding_model
        )
        logger.info('Saving Vector store')

        db.save_local(
            folder_path=DB_FAISS_PATH
        )

        logger.info('Vectorstore saved successfully')

        return db

    except Exception as e:
        error_message = CustomException('Failed to create new vector store',error_detail=e)
        logger.error(str(error_message))
        


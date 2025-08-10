from langchain_huggingface import HuggingFaceEmbeddings

from app.common.logger import get_logger

from app.common.custom_exception import CustomException

logger=get_logger(__name__)

def get_embedding_model():

    try:
        logger.info(f'Initialising our huggingface embedding model')

        embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiNiLM-L6-v2')

        logger.info(f'Hugging face embedding model loaded successfully')

        return embedding_model

    except Exception as e:
        error_message=CustomException('Error occured while loading model',error_detail=e)
        logger.error(str(error_message))
        raise error_message
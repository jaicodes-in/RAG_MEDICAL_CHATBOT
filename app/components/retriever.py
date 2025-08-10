from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from app.components.llm import load_llm

from app.components.vectorstore import load_vector_store

from app.config.config import HF_TOKEN,HUGGINGFACE_REPO_ID

from app.common.custom_exception import CustomException
from app.common.logger import get_logger

logger=get_logger(__name__)


CUSTOM_PROMPT_TEMPLATE="""
Answer the following medical questions in 2-3 lines
maximum using only the information provided in the context

##  context:
{context}

Question:
{question}

Answer:
"""

def set_custom_prompt():
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE,input_variables=['context','question'])


def create_qa_chain():
    try:
        logger.info('loading vector store for context')

        db=load_vector_store()

        llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID,hf_token=HF_TOKEN)

        if llm is None:
            raise CustomException("LLM not loaded")

        qa_chain=RetrievalQA.from_chain_type(
                llm=llm,
                chain_type='stuff',
                retriever=db.as_retriever(search_kwargs={'k':3}),
                return_source_documents=False,
                chain_type_kwargs={'prompt':set_custom_prompt()}
        )
        logger.info('Successfully created the QA chain')
        return qa_chain
    except Exception as e:
        error_message = CustomException('Failed to make a qa chain',error_detail=e)
        logger.error(str(error_message))

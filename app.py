from lcserve import serving
from typing import Union, List
from chain import get_content, get_qna_chain
from langchain import OpenAI
import dotenv

dotenv.load_dotenv()

def ask(url: Union[List[str], str], question: str) -> str:
    content = get_content(url)
    chain = get_qna_chain(OpenAI)
    return chain.run(input_document=content, question=question)
from lcserve import serving
from typing import Union, List
from chain import get_content, get_qna_chain
from langchain import OpenAI
from transcript import transcribe_video
from fastapi import FastAPI
import dotenv

app = FastAPI()



dotenv.load_dotenv()

def transcribe(vidurl: str) -> str:
    return transcribe_video(vidurl)

def ask(url: Union[List[str], str], question: str) -> str:
    content = get_content(url)
    chain = get_qna_chain(OpenAI)
    return chain.run(input_document=content, question=question)


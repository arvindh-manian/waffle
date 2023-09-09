from lcserve import serving
from typing import Union, List
from chain import get_content, get_qna_chain
from langchain import OpenAI
from transcript import transcribe_video
from fastapi import FastAPI
import dotenv

app = FastAPI()

dotenv.load_dotenv()


@app.get('/transcribe/')
def get_answer(url: str, question: str):
    transcription = transcribe_video(url)
    chain = get_qna_chain(OpenAI)
    return chain.run(input_document=transcription, question=question)
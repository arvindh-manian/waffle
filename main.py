from typing import Union, List
from chain import get_qna_chain
from langchain import OpenAI
from transcript import transcribe_video
from fastapi import FastAPI
import dotenv


dotenv.load_dotenv()
app = FastAPI()


@app.get('/')
def root():
    return "Hello World"

@app.get('/transcribe/')
def get_answer(url: str, question: str):
    transcription = transcribe_video(url)

    chain = get_qna_chain(OpenAI())
    return chain.run(input_document=transcription, question=question)


if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=rBR1DzRB0ss&pp=ygUFcG9rZXI%3D"
    question="What is the main concept?"

    print(get_answer(url, question))
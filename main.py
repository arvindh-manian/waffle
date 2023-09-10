from typing import Union, List
from chain import get_qna_chain, get_summary_chain, get_text_chunks_langchain
from langchain import OpenAI
from transcript import transcribe_video
from fastapi import FastAPI
from metaphor import find_links_for_question
import dotenv
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


class TranscriptionRequest(BaseModel):
    transcription: str
    question: str


dotenv.load_dotenv()
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_headers=["*"],
)



@app.get('/')
def root():
    return "Hello World"

@app.get("/transcribe_and_summarize/")
def get_answer(url: str):
    transcription = transcribe_video(url)

    chain = get_summary_chain(OpenAI())
    
    summary = chain.run(get_text_chunks_langchain(transcription))
    links = find_links_for_question(summary)

    return {
        "transcript": transcription,
        "summary": summary,
        "links": links
    }
    

@app.post("/ask/")
def get_answer(request: TranscriptionRequest):
    chain = get_qna_chain(OpenAI())
    
    answer = chain.run(input_document=request.transcription, question=request.question)
    
    return {
        "answer": answer
    }

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=rBR1DzRB0ss&pp=ygUFcG9rZXI%3D"
    question="What is the main concept?"

    print(get_answer(url, question))
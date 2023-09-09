from langchain.prompts.prompt import PromptTemplate
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.schema.document import Document

from typing import List, Union


from langchain.text_splitter import CharacterTextSplitter
def get_text_chunks_langchain(text):
   text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
   docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
   return docs

def get_content(filepath: str) -> str:
    with open(filepath, "r") as f:
        return f.read()

def get_combine_prompt() -> PromptTemplate:
    combine_prompt_template = """Given the following extracted parts of a video transcript and a question, generate an answer to the question based on information in the transcript. 
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.

    QUESTION: Which state/country's law governs the interpretation of the contract?
    =========
    Content: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in  relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an  injunction or other relief to protect its Intellectual Property Rights.

    Content: No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other)  right or remedy.\n\n11.7 Severability. The invalidity, illegality or unenforceability of any term (or part of a term) of this Agreement shall not affect the continuation  in force of the remainder of the term (if any) and this Agreement.\n\n11.8 No Agency. Except as expressly stated otherwise, nothing in this Agreement shall create an agency, partnership or joint venture of any  kind between the parties.\n\n11.9 No Third-Party Beneficiaries.

    Content: (b) if Google believes, in good faith, that the Distributor has violated or caused Google to violate any Anti-Bribery Laws (as  defined in Clause 8.5) or that such a violation is reasonably likely to occur,
    =========
    FINAL ANSWER: This Agreement is governed by English law.


    QUESTION: Where did Alfons Bürge get his PhD?
    =========
    Content: Alfons Bürge is a Swiss scholar of Ancient Law, with a special interest in the comparative study of Ancient and Modern Law.

    Content: Born in Winterthur, Switzerland, in 1947, Bürge studied the Classics at the University of Zurich. He received his Ph.D. in Classics from the University of Zurich in 1972 with a dissertation on the defense speech Pro Murena by Cicero (directed by Professor Heinz Haffter). 
    =========
    FINAL ANSWER: Alfons Bürge received his Ph.D. in Classics from the University of Zurich in 1972.
    
    
    QUESTION: {question}
    =========
    {summaries}
    =========
    FINAL ANSWER:"""

    return PromptTemplate(
        template=combine_prompt_template, input_variables=["summaries", "question"]
    )
    
def get_summary_chain(llm) -> AnalyzeDocumentChain:
    return load_summarize_chain(llm, chain_type="stuff")

def get_qna_chain(llm) -> AnalyzeDocumentChain:
    qa_chain = load_qa_chain(
        llm, chain_type="map_reduce", combine_prompt=get_combine_prompt()
    )

    return AnalyzeDocumentChain(combine_docs_chain=qa_chain)
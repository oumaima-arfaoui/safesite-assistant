import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from retriever import get_retriever

load_dotenv()
load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "nvidia/nemotron-3.5-lightning:free")
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "google/gemma-4-31b-it:free")


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata['source']}, p.{d.metadata['page']}]\n{d.page_content}"
        for d in docs
    )


PROMPT = ChatPromptTemplate.from_template(
    """You are SafeSite Assistant. Answer ONLY using the context below.
Cite your sources as [file, p.X], using the labels shown in the context.
If the context is not sufficient, reply exactly: "I could not find this in the safety documents."

Context:
{context}

Question: {question}"""
)

def make_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0,
        timeout=60,     
        max_retries=2,   
    )


llm = make_llm(LLM_MODEL).with_fallbacks([make_llm(LLM_FALLBACK_MODEL)])
retriever = get_retriever("similarity")

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)
answer_chain = PROMPT | llm | StrOutputParser()


def ask(question: str) -> tuple[str, list[Document]]:
    docs = retriever.invoke(question)
    answer = answer_chain.invoke(
        {"context": format_docs(docs), "question": question}
    )
    return answer, docs

if __name__ == "__main__":
    answer, docs = ask("Which PPE is required for concrete cutting?")
    print(answer)
    for d in docs:
        print(d.metadata["source"], d.metadata["page"])
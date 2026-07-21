"""
RAG chain using LangChain LCEL (LangChain 1.x compatible).
Retrieve relevant chunks from ChromaDB, then answer with GPT-4o-mini.
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "bsu_sfs"

SYSTEM_TEMPLATE = """You are a helpful assistant for Boise State University's Student Financial Services (SFS).
Answer questions using ONLY the context provided below. If the answer is not in the context, say:
"I don't have that specific information — please contact SFS directly at sfinfo@boisestate.edu or call (208) 426-1212."

Be concise, friendly, and accurate. Format dollar amounts and dates clearly.

Context:
{context}

Question: {question}
Answer:"""

_PROMPT = ChatPromptTemplate.from_template(SYSTEM_TEMPLATE)


def load_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def _format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_chain(vectorstore: Chroma):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20},
    )

    chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: _format_docs(retriever.invoke(x["question"]))),
            source_docs=RunnableLambda(lambda x: retriever.invoke(x["question"])),
        )
        | RunnablePassthrough.assign(answer=_PROMPT | llm | StrOutputParser())
    )
    return chain, retriever


def ask(chain_tuple, question: str) -> dict:
    chain, retriever = chain_tuple
    result = chain.invoke({"question": question})
    sources = list({
        doc.metadata.get("source", "")
        for doc in result.get("source_docs", [])
    })
    return {
        "answer": result["answer"],
        "sources": [s for s in sources if s],
    }

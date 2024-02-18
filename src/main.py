from config import MODEL_NAME, CHROMADB_DIR
from model import load_model
from helper_functions import format_docs, create_prompts, run_chat
from wiki_loader import load_wiki, vectorize_wiki
from rag_chain import create_rag_chain
from langchain_core.output_parsers import StrOutputParser

llm = load_model(MODEL_NAME)
contextualize_prompt, qa_prompt = create_prompts()

print("Пожалуйста, введите запрос для Википедии: ")
query = input()
data, all_splits = load_wiki(query, 'ru')
print("Результаты получены!")
print("Построение эмбеддингов...")
retriever = vectorize_wiki(all_splits, MODEL_NAME, CHROMADB_DIR)

contextualize_chain = contextualize_prompt | llm | StrOutputParser()

rag_chain = create_rag_chain(contextualize_chain, retriever, format_docs, qa_prompt, llm)

run_chat(rag_chain)

from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_wiki(query, lang):
    data = WikipediaLoader(
        query=query,
        load_max_docs=2,
        lang=lang,
        doc_content_chars_max=10000
    ).load()

    data_splits = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    ).split_documents(data)

    return data, data_splits


def vectorize_wiki(data_splits, model_name, chroma_dir):
    return Chroma.from_documents(
        documents=data_splits,
        embedding=LlamaCppEmbeddings(model_path='../models/' + model_name, verbose=False),
        persist_directory=chroma_dir
    ).as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )

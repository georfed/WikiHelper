from langchain_core.runnables import RunnablePassthrough


def create_rag_chain(contextualize_chain, retriever, format_docs, qa_prompt, llm):
    def contextualized_question(chain_input: dict):
        if chain_input.get('chat_history'):
            return contextualize_chain
        else:
            return chain_input['question']

    rag_chain = (
        RunnablePassthrough.assign(
            context=contextualized_question
                    | retriever
                    | format_docs
        )
        | qa_prompt
        | llm
    )

    return rag_chain

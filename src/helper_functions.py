from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config import CONTEXTUALIZE_PROMPT, QA_PROMPT


def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)


def create_prompts():
    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', CONTEXTUALIZE_PROMPT),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human', '{question}'),
        ]
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', QA_PROMPT),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human', '{question}'),
        ]
    )

    return contextualize_prompt, qa_prompt


def run_chat(rag_chain):
    chat_history = []
    print()
    print('Введите любой вопрос по заданной теме')

    while True:
        question = input('Вопрос: ')
        print('Пожалуйста, подождите')
        context = {'question': question, 'chat_history': chat_history}
        ai_msg = rag_chain.invoke(context)
        chat_history.extend([
            ChatMessage(role='User', content=question),
            ChatMessage(role='Assistant', content=ai_msg),
        ])
        chat_history = chat_history[-4:]
        print(f'Ответ: {ai_msg}')
        print()

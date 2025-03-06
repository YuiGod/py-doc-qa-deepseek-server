from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from models.chat_history_model import ChatHistory
from .base import COLLECTION_NAME, MODEL_NAME, VECTOR_DIR


def build_history_template(chat_history_list: list[ChatHistory]):
    """构建聊天历史模板"""

    if type(chat_history_list) != list or len(chat_history_list) == 0:
        return []
        # raise HTTPException(status_code=500, detail="没有对话消息列表。")

    history_messages: list[BaseMessage] = []
    # 历史记录转换为 LangChain 消息对象
    for history in chat_history_list:
        if history.role == "user":
            history_messages.append(HumanMessage(content=history.content))
        elif history.role == "assistant":
            history_messages.append(AIMessage(content=history.content))
    return history_messages


def build_qa_chain():

    vector_store = Chroma(
        persist_directory=VECTOR_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=OllamaEmbeddings(model=MODEL_NAME),
    )

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0.3,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20,
            "lambda_mult": 0.5,
            "score_threshold": 0.6,
        },
    )

    # system 提示词模板
    system_template = """
        您是一个设计用于査询文档来回答问题的代理，您的名字是超级牛逼哄哄的小天才助手。
        您可以使用文档检索工具，并基于检索内容来回答问题。
        您可能不查询文档就知道答案，但是您仍然应该查询文档来获得答案。
        如果用户的问题与检索文档上下文的内容无关，您仍然应该查询文档来获得答案。
        如果您从文档中找不到任何信息用于回答问题，则只需返回“抱歉，这个问题我还不知道。”作为答案。
        文档上下文：{context}
        """
    prompt = ChatPromptTemplate(
        [
            ("system", system_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )

    return (
        {
            "question": lambda x: x["question"],
            "context": lambda x: retriever.invoke(x["question"]),
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

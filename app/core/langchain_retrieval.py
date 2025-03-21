from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from models.chat_history_model import ChatHistory
from .base import chat_llm, chroma_vector_store


def build_history_template(chat_history_list: list[ChatHistory]):
    """构建聊天历史模板"""

    if type(chat_history_list) != list or len(chat_history_list) == 0:
        return []

    history_messages: list[BaseMessage] = []
    # 历史记录转换为 LangChain 消息对象数组
    for history in chat_history_list:
        if history.role == "user":
            history_messages.append(HumanMessage(content=history.content))
        elif history.role == "assistant":
            history_messages.append(AIMessage(content=history.content))
    return history_messages


def build_qa_chain():

    # 初始化 Chroma 向量数据库
    vector_store = chroma_vector_store()

    # 初始化 deepseek 模型
    llm = chat_llm()

    # 初始化检索，并配置
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,  # 检索结果返回最相似的文档数量
            "fetch_k": 20,  # 要传递给 MMR 算法的文档量
            "lambda_mult": 0.5,  # MMR 返回的结果多样性，1 表示最小多样性，0 表示最大值。（默认值：0.5）
        },
    )

    # system 提示词模板
    system_template = """
        您是一个设计用于査询文档来回答问题的代理，您的名字是超级牛逼哄哄的小天才助手。
        您可以使用文档检索工具，并基于检索内容来回答问题。不需要说出检索文档的id。
        您可能不查询文档就知道答案，但是您仍然应该查询文档来获得答案。
        如果用户的问题与检索文档上下文的内容无关，您仍然应该查询文档来获得答案。
        结合文档内容作为答案,切记不可自由发挥,不可胡编乱造。
        如果您从文档中找不到任何信息用于回答问题，则只需返回“抱歉，这个问题我还不知道。”作为答案。
        文档内容：{context}
        """
    prompt = ChatPromptTemplate(
        [
            ("system", system_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )

    # 构建检索链管道 Runnable
    # retriever.invoke() 作用是根据用户问题检索匹配最相关的文档
    # x 值是管道里的参数，包括 question，chat_history，还要其他有关langchain的参数
    return (
        {
            "context": lambda x: retriever.invoke(x["question"]),
            "chat_history": lambda x: x["chat_history"],
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

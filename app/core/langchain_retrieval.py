from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import readline

VECTOR_DIR = "/home/ly/Project/vector_store"
MODEL_NAME = "deepseek-r1:7b"

# 会话记忆缓冲区，用于存储对话历史，保存在内存中
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")


def build_qa_chain():

    vector_store = Chroma(
        persist_directory=VECTOR_DIR,
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

    system_template = """
        您是一名超级牛逼哄哄的小天才助手，是一个设计用于査询文档来回答问题的代理。
        您可以使用文档检索工具，并基于检索内容来回答问题。
        您可能不查询文档就知道答案，但是您仍然应该查询文档来获得答案。
        如果您从文档中找不到任何信息用于回答问题，则只需返回“抱歉，这个问题我还不知道。”作为答案。
        如果有人提问等关于您的名字的问题，您就回答：“我是超级牛逼哄哄的小天才助手”作为答案。
        上下文：{context}
        """
    prompt = ChatPromptTemplate(
        [
            ("system", system_template),
            MessagesPlaceholder("chat_history"),  # 将历史对话插入到模板中
            ("human", "{question}"),
        ]
    )

    return (
        {
            "question": RunnablePassthrough(),
            "context": retriever,
            "chat_history": lambda x: memory.load_memory_variables({})["chat_history"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def console_qa():
    print("初始化知识库系统...")
    chain = build_qa_chain()
    # 交互界面
    print("系统就绪，输入问题开始对话（输入 'exit' 退出）")
    while True:
        try:
            query = input("\n问题：").strip()
            if not query or query.lower() in ("exit", "quit"):
                break

            print("回答：", end="", flush=True)
            response = ""

            for chunk in chain.invoke(query):
                response += chunk

            # 截取 </think> 后面的字符串
            split_string = lambda str: (
                str.split("</think>", 1)[1] if "</think>" in str else str
            )
            # 更新记忆
            memory.save_context({"inputs": query}, {"outputs": split_string(response)})

            print("\n\n")
            print("==== 请继续对话（输入 'exit' 退出）====")

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    console_qa()
    print("对话结束")

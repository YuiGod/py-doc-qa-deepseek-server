from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader
import os
import shutil

"""
超简单的 RAG 示例代码
"""

# 指定持久化向量数据库的存储路径
VECTOR_DIR = "/home/ly/Project/vector_store"
# 模型名称
MODEL_NAME = "deepseek-r1:7b"

# 1. 加载文档（示例使用单个文本文件）
loader = TextLoader("/home/ly/Project/fileStorage/FFF团.txt")
documents = loader.load()

# 2. 文本分割（优化中文处理）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", "。", "；", " ", ""]
)
split_docs = text_splitter.split_documents(documents)
print(f"分割文档数: {len(split_docs)}")

# 3. 初始化模型
embeddings = OllamaEmbeddings(model=MODEL_NAME)
llm = ChatOllama(model=MODEL_NAME, temperature=0.3)

# 4. 创建向量数据库（自动持久化）
# 清理旧向量数据库的数据
if os.path.exists(VECTOR_DIR):
    shutil.rmtree(VECTOR_DIR)

vector_db = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory=VECTOR_DIR,
)
print("向量数据库已创建")

# 5. 构建检索链（优化中文模板）

prompt_template = """
您是一个设计用于査询文档来回答问题的代理。您可以使用文档检索工具。
并基于检索内容来回答问题您可能不查询文档就知道答案，但是您仍然应该查询文档来获得答案。
如果您从文档中找不到任何信息用于回答问题，则只需返回“抱歉，这个问题我还不知道。”作为答案。
如果有人提问等关于您的名字的问题，您就回答：“我是超级牛逼哄哄的小天才助手”作为答案。
上下文：{context}
问题：{question}
"""

retriever = vector_db.as_retriever(
    search_type="mmr",  # 最大边际相关性搜索（平衡相关性与多样性）
    search_kwargs={
        "k": 5,  # 初始检索文档数量
        "fetch_k": 20,  # 底层搜索数量（越大精度越高）
        "lambda_mult": 0.5,  # 多样性控制参数（0-1，越大越多样）
        "score_threshold": 0.3,  # 相关性阈值
    },
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | ChatPromptTemplate.from_template(prompt_template)
    | llm
)

# 6. 执行查询
response = chain.invoke("FFF团的惩罚是什么？")
print(f"\n回答：\n{response.content}")

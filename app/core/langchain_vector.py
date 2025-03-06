from fastapi import HTTPException
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time

from .base import COLLECTION_NAME, LOAD_PATH, MODEL_NAME, VECTOR_DIR


def load_documents(source_dir=LOAD_PATH):
    """
    加载指定目录下的所有文档
    支持格式：.txt, .pdf, .docx, .md
    """

    try:
        # 分别加载不同格式
        text_loader = DirectoryLoader(
            path=source_dir,  # 指定读取文件的父目录
            glob=["**/*.txt", "**/*.md"],  # 指定读取文件的格式
            show_progress=True,  # 显示加载进度
            use_multithreading=True,  # 使用多线程
            loader_cls=TextLoader,  # 指定加载器
            loader_kwargs={"autodetect_encoding": True},  # 自动检测文件编码
        )

        pdf_loader = DirectoryLoader(
            path=source_dir,
            glob="**/*.pdf",
            show_progress=True,
            use_multithreading=True,
            loader_cls=PyPDFLoader,
        )

        docx_loader = DirectoryLoader(
            path=source_dir,
            glob="**/*.docx",
            show_progress=True,
            use_multithreading=True,
            loader_cls=Docx2txtLoader,
            loader_kwargs={"autodetect_encoding": True},
        )
        # 合并文档列表
        docs = []
        docs.extend(text_loader.load())
        docs.extend(pdf_loader.load())
        docs.extend(docx_loader.load())
        print(f"成功加载 {len(docs)} 份文档")
        return docs
    except Exception as e:
        print(f"加载文档失败：{str(e)}")
        raise HTTPException(status_code=500, detail=f"加载文档失败：{str(e)}")


def split_documents(documents, chunk_size=800, chunk_overlap=150):
    """
    使用递归字符分割器处理文本
    参数说明：
    - chunk_size：每个文本块的最大字符数，推荐 500-1000
    - chunk_overlap：相邻块之间的重叠字符数（保持上下文连贯），推荐 100-200
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "。", "!", "?", "？", "！", "；", ";"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,  # 保留原始文档中的位置信息
    )

    split_docs = text_splitter.split_documents(documents)
    print(f"原始文档数：{len(documents)}")
    print(f"分割后文本块数：{len(split_docs)}")

    return split_docs


def create_vector_store(split_docs, persist_dir=VECTOR_DIR):
    """
    创建持久化向量数据库
    - split_docs: 经过分割的文档列表
    - persist_dir: 向量数据库存储路径（建议使用WSL原生路径）
    """

    # 初始化 Chroma 向量数据库
    vector_store = Chroma(
        persist_directory=persist_dir,
        collection_name=COLLECTION_NAME,
        embedding_function=OllamaEmbeddings(model=MODEL_NAME),
    )

    # 向量化文档之前，先把原来集合里的数据清空
    ids = vector_store._collection.get()["ids"]
    if len(ids):
        vector_store.delete(ids=vector_store._collection.get()["ids"])

    # 如果分割文档为空，不做向量化操作
    if not split_docs or len(split_docs) == 0:
        return

    try:
        start_time = time.time()
        print(f"\n开始向量化====>")

        # vector_store = Chroma.from_documents(
        #     client=chroma_client,
        #     collection_name="documents_qa",
        #     documents=split_docs,
        #     embedding=OllamaEmbeddings(model=MODEL_NAME),
        # )
        # 向量化文档到向量数据库
        vector_store.add_documents(split_docs)

        print(f"\n向量化完成！耗时 {time.time()-start_time:.2f} 秒")
        print(f"数据库存储路径：{persist_dir}")
        print(f"总文档块数：{vector_store._collection.count()}")

    except Exception as e:
        print(f"向量化失败：{str(e)}")
        raise HTTPException(status_code=500, detail=f"向量化失败：{str(e)}")


def vector_documents():
    """
    启动文档向量化，并保存数据库
    """
    # 加载本地文档
    documents = load_documents()
    # 执行分割
    split_docs = split_documents(documents)
    # 执行向量化（使用之前分割好的split_docs）
    create_vector_store(split_docs)

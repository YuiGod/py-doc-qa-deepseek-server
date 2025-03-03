from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import time

"""
LangChain 文本加载，切割，向量化保存向量数据库 示例代码
"""

# 指定加载文档的目录
LOAD_PATH = "/home/ly/Project/fileStorage"
# 指定持久化向量数据库的存储路径
VECTOR_DIR = "/home/ly/Project/vector_store"


def load_documents(source_dir: str):
    """
    加载指定目录下的所有文档
    支持格式：.txt, .pdf, .docx, .md
    """

    # 分别加载不同格式
    text_loader = DirectoryLoader(
        path=source_dir,  # 指定读取文件的父目录
        glob=["**/*.txt", "**/*.md"],  # 指定读取文件的格式
        show_progress=True,  # 显示加载进度
        use_multithreading=True,  # 使用多线程
        # silent_errors=True,  # 错误时不抛出异常，直接忽略该文件
        loader_cls=TextLoader,  # 指定加载器
        loader_kwargs={"autodetect_encoding": True},  # 自动检测文件编码
    )

    pdf_loader = DirectoryLoader(
        path=source_dir,
        glob="**/*.pdf",
        show_progress=True,
        use_multithreading=True,
        # silent_errors=True,
        loader_cls=PyPDFLoader,
        loader_kwargs={"autodetect_encoding": True},
    )

    docx_loader = DirectoryLoader(
        path=source_dir,
        glob="**/*.docx",
        show_progress=True,
        use_multithreading=True,
        # silent_errors=True,
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

    # 查看分割效果示例
    print("\n示例文本块：")
    print(split_docs[0].page_content[:300] + "...")
    print(f"元数据：{split_docs[0].metadata}")

    return split_docs


def create_vector_store(split_docs, persist_dir=VECTOR_DIR):
    """
    创建持久化向量数据库
    :param split_docs: 经过分割的文档列表
    :param persist_dir: 向量数据库存储路径（建议使用WSL原生路径）
    """

    # 初始化本地嵌入模型
    embeddings = OllamaEmbeddings(model="deepseek-r1:7b")

    try:
        start_time = time.time()

        # 创建带进度显示的向量数据库
        db = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=persist_dir,  # 持久化存储路径
        )

        print(f"\n向量化完成！耗时 {time.time()-start_time:.2f} 秒")
        print(f"数据库存储路径：{persist_dir}")
        print(f"总文档块数：{db._collection.count()}")

        return db
    except Exception as e:
        print(f"向量化失败：{str(e)}")
        return None


# 加载本地文档
documents = load_documents(LOAD_PATH)
# 测试是否成功加载文档
for doc in documents[:2]:  # 打印前两篇摘要
    print(f"文件路径: {doc.metadata['source']}")
    print(f"内容预览: {doc.page_content[:150]}...\n")
# 执行分割
split_docs = split_documents(documents)
# 执行向量化（使用之前分割好的split_docs）
vector_db = create_vector_store(split_docs)

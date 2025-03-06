from typing import Annotated
from fastapi import Depends
from sqlmodel import SQLModel, Session, create_engine


# 导入所有数据库表
from models import document_model, chat_session_model, chat_history_model

# 创建数据库
sqlite_file_name = "document_qa.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

engine = create_engine(sqlite_url, echo=True)


def create_db_and_tables():
    """创建数据库和所有表"""
    SQLModel.metadata.create_all(engine)


def get_session():
    """依赖注入，将engine注册到FastApi session中"""
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]

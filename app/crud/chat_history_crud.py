from sqlmodel import Session, desc, select
from models.chat_history_model import ChatHistory, ChatHistoryCreate
from models.chat_model import Chatting

from .base import engine


class ChatHistoryCrud:

    def add_item(slef, chatHistory: ChatHistoryCreate):
        """chat历史添加记录"""
        chatHistory = chatHistory.model_dump(exclude_unset=True)

        with Session(engine) as session:
            db_history = ChatHistory.model_validate(chatHistory)
            session.add(db_history)
            session.commit()
            session.refresh(db_history)
            return db_history

    def list_by_chat_session_id(self, chatSessionId: str):
        with Session(engine) as session:
            query = (
                select(ChatHistory)
                .where(ChatHistory.chat_session_id == chatSessionId)
                .order_by(ChatHistory.date)
            )
            resultList = session.exec(query).all()
            return resultList

    def delete_by_chat_session_id(self, chatSessionId: str):
        """删除历史记录"""
        with Session(engine) as session:
            query = select(ChatHistory).where(
                ChatHistory.chat_session_id == chatSessionId
            )

            resultList = session.exec(query).all()
            for dbChat in resultList:
                session.delete(dbChat)
            session.commit()

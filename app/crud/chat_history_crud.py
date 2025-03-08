from sqlmodel import Session, desc, select
from models.chat_history_model import ChatHistory, ChatHistoryCreate

from .base import engine


class ChatHistoryCrud:

    def add_item(slef, chat_history: ChatHistoryCreate):
        """chat历史添加记录"""
        chat_history = chat_history.model_dump(exclude_unset=True)

        with Session(engine) as session:
            db_history = ChatHistory.model_validate(chat_history)
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
            result_list = session.exec(query).all()
            return result_list

    def delete_by_chat_session_id(self, chatSessionId: str):
        """删除历史记录"""
        with Session(engine) as session:
            query = select(ChatHistory).where(
                ChatHistory.chat_session_id == chatSessionId
            )

            result_list = session.exec(query).all()
            for db_chat in result_list:
                session.delete(db_chat)
            session.commit()

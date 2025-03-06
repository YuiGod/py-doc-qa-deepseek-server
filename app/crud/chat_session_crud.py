from fastapi import HTTPException
from sqlmodel import Session, desc, select
from models.chat_session_model import ChatSession, ChatSessionParams, ChatSessionUpdate

from .base import engine


class ChatSessionCrud:
    def save(self, data: ChatSessionParams):

        chatSession = ChatSessionUpdate(title=data.title)

        if data.id:
            with Session(engine) as session:
                dbUpdateSession = session.get(ChatSession, data.id)
                chatSession = chatSession.model_dump(exclude_unset=True)
                dbUpdateSession.sqlmodel_update(chatSession)
                session.add(dbUpdateSession)
                session.commit()
                session.refresh(dbUpdateSession)
                return dbUpdateSession

        with Session(engine) as session:
            dbAddSession = ChatSession.model_validate(chatSession)
            session.add(dbAddSession)
            session.commit()
            session.refresh(dbAddSession)
            return dbAddSession

    def list(self):
        with Session(engine) as session:
            query = select(ChatSession).order_by(desc(ChatSession.date))
            chatSessionList = session.exec(query).all()
            return chatSessionList

    def delete(self, id: str):
        """删除会话记录"""
        with Session(engine) as session:
            db = session.get(ChatSession, id)
            if not db:
                raise HTTPException(status_code=500, detail="会话记录不存在。")

            session.delete(db)
            session.commit()

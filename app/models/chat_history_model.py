from datetime import datetime
import uuid
from pydantic import BaseModel, field_validator
from sqlmodel import Field, SQLModel


class ChatHistory(SQLModel, table=True):
    """chathistory表"""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    role: str
    content: str
    think: str | None = None
    chat_session_id: uuid.UUID | None = None
    date: datetime = Field(default_factory=datetime.now)


class ChatHistoryCreate(SQLModel):
    role: str
    content: str
    think: str | None = None
    chat_session_id: uuid.UUID | None = None


class ChatHistoryFormat(SQLModel):
    """对话记录（格式化日期）"""

    id: uuid.UUID
    role: str
    content: str
    think: str | None = None
    chat_session_id: uuid.UUID | None = None
    date: str

    @field_validator("date", mode="before")
    @classmethod
    def format_date_v2(cls, value):
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d %H:%M:%S")
        return value


class ChatHistoryResponse(BaseModel):
    """响应体"""

    code: int
    message: str
    data: list[ChatHistoryFormat]

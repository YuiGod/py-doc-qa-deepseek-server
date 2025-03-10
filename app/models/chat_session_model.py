from datetime import datetime
from typing import Optional
import uuid
from pydantic import BaseModel, field_validator
from sqlmodel import Field, SQLModel


class ChatSession(SQLModel, table=True):
    """chatsession表"""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    title: str | None = None
    date: datetime = Field(default_factory=datetime.now)


class ChatSessionUpdate(SQLModel):
    """对话保存更新Model"""

    title: Optional[str] = None


class ChatSessionParams(BaseModel):
    """对话请求参数"""

    id: Optional[uuid.UUID] = None
    title: Optional[str] = None


class ChatSessionFormat(SQLModel):
    """对话记录（格式化日期）"""

    id: uuid.UUID
    title: str | None = None
    date: str

    @field_validator("date", mode="before")
    @classmethod
    def format_date_v2(cls, value):
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d %H:%M:%S")
        return value


class ChatSessionResponse(BaseModel):
    """响应体"""

    code: int
    message: str
    data: list[ChatSessionFormat]

import uuid
from datetime import datetime
from typing import Optional
from fastapi import File, UploadFile
from sqlmodel import Field, SQLModel
from pydantic import BaseModel, field_validator


class Document(SQLModel, table=True):
    """document表"""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str
    file_name: str = Field(index=True)
    file_path: str | None = None
    suffix: str | None = None
    vector: str | None = None
    date: datetime = Field(default_factory=datetime.now)


class DocumentUpdate(SQLModel):
    name: Optional[str] = None
    file_name: Optional[str] = None
    file_path: Optional[str] = None
    suffix: Optional[str] = None
    vector: Optional[str] = None


class UploadFormData(SQLModel):
    """上传文档数据，附带表单数据"""

    id: Optional[str] = None
    name: str
    file_name: Optional[str] = None
    file_path: Optional[str] = None
    suffix: Optional[str] = None
    vector: Optional[str] = None
    date: Optional[str] = None
    file: UploadFile = File()


class UpdateFormData(UploadFormData):
    """修改上传文档数据，附带表单数据"""

    id: Optional[uuid.UUID] = None
    file: Optional[UploadFile] = None


class DocumentParams(SQLModel):
    """请求参数"""

    id: Optional[uuid.UUID] = None
    name: Optional[str] = None
    file_name: Optional[str] = None
    file_path: Optional[str] = None
    suffix: Optional[str] = None
    vector: Optional[str] = None
    date: Optional[str] = None
    page_num: Optional[int] = None
    page_size: Optional[int] = None


class DocumentFormat(SQLModel):
    """文档记录（格式化日期）"""

    id: uuid.UUID
    name: str
    file_name: str
    file_path: str | None = None
    suffix: Optional[str] = None
    vector: str | None = None
    date: str

    @field_validator("date", mode="before")
    @classmethod
    def format_date_v2(cls, value):
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d %H:%M:%S")
        return value


class DocumentPage(BaseModel):
    """文档分页数据"""

    total: int
    page_num: int
    page_size: int
    list: list[DocumentFormat]


class DocumentResponse(BaseModel):
    """响应体"""

    code: int
    message: str
    data: DocumentPage

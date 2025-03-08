from typing import Annotated
import uuid
from fastapi import APIRouter, Form, Query
from fastapi.responses import FileResponse
from core.langchain_vector import vector_documents
from crud.document_crud import DocumentCrud
from models.document_model import (
    DocumentParams,
    DocumentResponse,
    UpdateFormData,
    UploadFormData,
)
from urllib.parse import quote
from routers.base import success


router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    responses={404: {"message": "您所访问的资源不存在！"}},
)

document_crud = DocumentCrud()


@router.post("/add")
async def add_doc(data: Annotated[UploadFormData, Form()]):
    await document_crud.add(data)
    return success(None, "添加成功！")


@router.put("/update")
async def update_doc(data: Annotated[UpdateFormData, Form()]):
    await document_crud.update(data)
    return success(None, "更新成功！")


@router.get("/page", response_model=DocumentResponse)
async def page_doc(params: Annotated[DocumentParams, Query()]):
    result = document_crud.page(params)
    return success(result)


@router.delete("/delete")
async def del_doc(data: DocumentParams):
    await document_crud.delete(data.id)
    return success(None, "删除成功！")


@router.get("/read/{item_id}")
async def read_doc_file(item_id: uuid.UUID):
    file_path, real_name = document_crud.download(item_id)
    header_file_name = quote(real_name, encoding="utf-8")
    headers = {"Content-Disposition": f"inline; filename*=UTF-8''{header_file_name}"}
    return FileResponse(path=file_path, headers=headers, media_type=None)


@router.get("/vector-all")
async def vector_docs():
    vector_documents()
    document_crud.vector_all_docs()
    return success(None, "已全部向量化。")

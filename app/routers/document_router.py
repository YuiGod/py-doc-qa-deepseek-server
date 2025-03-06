from typing import Annotated
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
from routers.base import success


router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    responses={404: {"message": "您所访问的资源不存在！"}},
)

document_crud = DocumentCrud()


@router.post("/addDoc")
async def add_doc(data: Annotated[UploadFormData, Form()]):
    await document_crud.add(data)
    return success(None, "保存成功！")


@router.put("/editDoc")
async def edit_doc(data: Annotated[UpdateFormData, Form()]):
    await document_crud.update(data)
    return success(None, "更新成功！")


@router.get("/pageDoc", response_model=DocumentResponse)
async def page_doc(params: Annotated[DocumentParams, Query()]):
    result = document_crud.page(params)
    return success(result)


@router.delete("/delDoc")
async def del_doc(data: DocumentParams):
    await document_crud.delete(data.id)
    return success(None, "删除成功！")


@router.post("/read")
async def read_doc_file(data: DocumentParams):
    file_path, realName = document_crud.download(data.id)
    headers = {"Content-Disposition": f"inline; filename*=UTF-8''{realName}"}
    return FileResponse(path=file_path, headers=headers, media_type=None)


@router.get("/vector-all")
async def vector_docs():
    vector_documents()
    document_crud.vector_all_docs()
    return success(None, "已全部向量化。")

from fastapi import APIRouter

from crud.chat_history_crud import ChatHistoryCrud
from crud.chat_session_crud import ChatSessionCrud
from models.chat_session_model import ChatSessionParams, ChatSessionResponse
from routers.base import success


router = APIRouter(
    prefix="/session",
    tags=["session"],
    responses={404: {"message": "您所访问的资源不存在！"}},
)

chatSessionCrud = ChatSessionCrud()
chatHistoryCrud = ChatHistoryCrud()


@router.get("/list", response_model=ChatSessionResponse)
async def chat_session_list():
    results = chatSessionCrud.list()
    return success(results)


@router.post("/add")
async def chat_session_add(params: ChatSessionParams):
    results = chatSessionCrud.save(params)
    return success(results)


@router.put("/update")
async def chat_session_add(params: ChatSessionParams):
    results = chatSessionCrud.save(params)
    return success(results, "修改成功！")


@router.delete("/delete")
async def chat_session_add(params: ChatSessionParams):
    chatHistoryCrud.delete_by_chat_session_id(params.id)
    chatSessionCrud.delete(params.id)
    return success(None, "删除成功！")

import json
import time
from typing import Annotated
from fastapi import APIRouter, Form, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from models.chat_history_model import ChatHistoryCreate
from crud.chat_history_crud import ChatHistoryCrud
from models.chat_session_model import ChatSessionParams, ChatSessionResponse
from core.langchain_retrieval import build_history_template, build_qa_chain
from models.chat_model import ChatParams, ChatStreamResponse, Chatting
from routers.base import success


router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    responses={404: {"message": "您所访问的资源不存在！"}},
)

chatHistoryCrud = ChatHistoryCrud()


@router.post("")
async def chatting(data: ChatParams):
    if not data.messages:
        raise HTTPException(status_code=500, detail="网络异常，请稍后重试！")

    # 获取历史记录
    historyList = chatHistoryCrud.list_by_chat_session_id(data.chat_session_id)
    # 历史记录转换成LangChain提示词模板
    historyPrompt = build_history_template(historyList)
    # LangChain 检索链 astream() 的参数
    invokeParams = {"question": data.messages.content, "chat_history": historyPrompt}

    # user 消息添加到历史记录中
    userChat = ChatHistoryCreate(
        role=data.messages.role,
        content=data.messages.content,
        chat_session_id=data.chat_session_id,
    )
    chatHistoryCrud.add_item(userChat)

    try:
        chain = build_qa_chain()
        return StreamingResponse(
            generate_stream(chain, invokeParams, data.chat_session_id),
            media_type="application/x-ndjson",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"流式响应失败：{str(e)}")


# chat 返回响应流
async def generate_stream(chain, invokeParams, chat_session_id):
    """LangChain 流响应转 JSON 字符串流响应"""

    responseChat = ""
    async for chunk in chain.astream(invokeParams):
        responseChat += chunk

        json_chunk = json.dumps(
            jsonable_encoder(
                ChatStreamResponse(
                    model="deepseek-r1:7b",
                    created_at=int(round(time.time() * 1000)),
                    message=Chatting(role="assistant", content=chunk),
                    done=False,
                ).model_dump(exclude_none=True)
            )
        )
        # 换行符分隔JSON行
        yield f"{json_chunk}\n"

    # 流结束后发送完成标记
    done = json.dumps(
        jsonable_encoder(
            ChatStreamResponse(
                model="deepseek-r1:7b",
                created_at=int(round(time.time() * 1000)),
                message=Chatting(role="assistant", content=""),
                done=True,
                done_reason="stop",
            )
        )
    )
    yield f"{done}\n"

    # 流式响应完成后，assistant 消息保存到历史消息记录中
    think = ""
    content = ""
    if "<think>" in responseChat:
        think = responseChat.split("<think>")[1].split("</think>")[0]
        content = responseChat.split("</think>")[1]
    else:
        content = responseChat

    assistantChat = ChatHistoryCreate(
        role="assistant",
        content=content,
        think=think,
        chat_session_id=chat_session_id,
    )
    chatHistoryCrud.add_item(assistantChat)


@router.get("/history")
async def chat_history(params: Annotated[ChatSessionParams, Query()]):
    results = chatHistoryCrud.list_by_chat_session_id(params.id)
    return success(results)

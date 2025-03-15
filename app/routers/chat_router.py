import json
import time
from typing import Annotated
from fastapi import APIRouter, Form, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from crud.chat_history_crud import ChatHistoryCrud
from models.chat_history_model import ChatHistoryCreate, ChatHistoryResponse
from models.chat_session_model import ChatSessionParams
from models.chat_model import ChatParams, ChatStreamResponse, Chatting
from core.base import MODEL_NAME
from core.langchain_retrieval import build_history_template, build_qa_chain
from routers.base import success


router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    responses={404: {"message": "您所访问的资源不存在！"}},
)

chat_history_crud = ChatHistoryCrud()


@router.post("")
async def chatting(data: ChatParams):
    if not data.messages:
        raise HTTPException(status_code=500, detail="网络异常，请稍后重试！")

    # 先获取历史记录
    history_list = chat_history_crud.list_by_chat_session_id(data.chat_session_id)
    # 再保存 user 消息到历史记录中
    user_chat = ChatHistoryCreate(
        role=data.messages.role,
        content=data.messages.content,
        chat_session_id=data.chat_session_id,
    )
    chat_history_crud.add_item(user_chat)

    # 历史记录转换成LangChain提示词模板
    history_message = build_history_template(history_list)
    # LangChain 检索链 astream() 的参数
    invoke_params = {"question": data.messages.content, "chat_history": history_message}

    try:
        chain = build_qa_chain()
        return StreamingResponse(
            generate_stream(chain, invoke_params, data.chat_session_id),
            media_type="application/x-ndjson",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"流式响应失败：{str(e)}")


# chat 返回响应流
async def generate_stream(chain, invoke_params, chat_session_id):
    """LangChain 流响应转 JSON 字符串流响应"""

    think = ""
    content = ""
    isThinking = False
    # 一个跳过本次循环的标记，目的是剔除think标签
    loop_continue = False

    async for chunk in chain.astream(invoke_params):
        # 收集 think 和 content
        loop_continue = False
        if "<think>" in chunk:
            isThinking = True
            loop_continue = True
        if "</think>" in chunk:
            isThinking = False
            loop_continue = True
        if not loop_continue:
            if isThinking:
                think += chunk
            else:
                content += chunk

        json_chunk = json.dumps(
            jsonable_encoder(
                ChatStreamResponse(
                    model=MODEL_NAME,
                    created_at=int(round(time.time() * 1000)),
                    message=Chatting(role="assistant", content=chunk),
                    done=False,
                ).model_dump(exclude_none=True)
            ),
            ensure_ascii=False,
        )
        # 换行符分隔JSON行
        yield f"{json_chunk}\n"

    # 流结束后发送完成标记
    done = json.dumps(
        jsonable_encoder(
            ChatStreamResponse(
                model=MODEL_NAME,
                created_at=int(round(time.time() * 1000)),
                message=Chatting(role="assistant", content=""),
                done=True,
                done_reason="stop",
            )
        ),
        ensure_ascii=False,
    )
    yield f"{done}\n"

    # 流式响应完成后，assistant 消息保存到历史消息记录中
    assistantChat = ChatHistoryCreate(
        role="assistant",
        content=content,
        think=think,
        chat_session_id=chat_session_id,
    )
    chat_history_crud.add_item(assistantChat)


@router.get("/history", response_model=ChatHistoryResponse)
async def chat_history(params: Annotated[ChatSessionParams, Query()]):
    results = chat_history_crud.list_by_chat_session_id(params.id)
    return success(results)

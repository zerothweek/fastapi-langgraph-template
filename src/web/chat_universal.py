from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from model.chat_universal import(
    ChatUniversalRequest
)
from service.chat_universal import ChatUniversalService

router = APIRouter(prefix="/chat-universal")
service = ChatUniversalService()

@router.post("chat_stream")
async def general_chat_stream(req: ChatUniversalRequest) -> StreamingResponse:
    async def gen():
        async for chunk in service.stream_chat(
            thread_id=req.thread_id,
            message=req.message,
        ):
            yield chunk
    
    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    })
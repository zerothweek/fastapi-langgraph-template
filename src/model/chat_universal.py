from pydantic import BaseModel, Field

class ChatUniversalRequest(BaseModel):
    thread_id: str | None = Field(default=None, description="Thread id for multi turn")
    message: str


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from src.agent.llm_agent import create_running_coach_agent


app = FastAPI(
    title="Agentic RAG Running Coach API",
    description="FastAPI backend for Running Coach & Shoe Recommender Agent",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = create_running_coach_agent()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    chat_history: Optional[List[ChatMessage]] = []


class ChatResponse(BaseModel):
    answer: str
    response_mode: Optional[str] = None
    action_plan: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = []
    retrieved_coach_docs: Optional[List[Dict[str, Any]]] = []
    retrieved_shoe_docs: Optional[List[Dict[str, Any]]] = []
    constraints: Optional[Dict[str, Any]] = {}
    note: Optional[str] = None


def build_chat_history_text(history: List[ChatMessage]) -> str:
    if not history:
        return ""

    lines = []
    for msg in history[-8:]:
        role = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{role}: {msg.content}")

    return "\n".join(lines)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Running Coach API is ready"
    }


@app.get("/sample-queries")
def sample_queries():
    return {
        "queries": [
            "อยากทำ 10K sub-55 ตอนนี้วิ่งได้ 60 นาที มีเวลาซ้อม 3 วัน/สัปดาห์ ภายใน 8 สัปดาห์",
            "อยากวิ่ง 5K ให้ได้ 30 นาที ตอนนี้ 40 นาที ซ้อมได้ 4 วัน/สัปดาห์",
            "ช่วยแนะนำรองเท้าวิ่งสำหรับ easy run งบไม่เกิน 4000 บาท",
            "ช่วยแนะนำรองเท้าวิ่งสำหรับ easy run งบไม่เกิน 4000 บาท ของ nike หน่อย",
            "อยากวิ่ง 5K ให้ได้ 30 นาที ตอนนี้ 40 นาที ซ้อมได้ 4 วัน/สัปดาห์ และแนะนำรองเท้าสำหรับ tempo run ให้หน่อย"
        ]
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    chat_history_text = build_chat_history_text(request.chat_history or [])

    result = agent.invoke({
        "input": request.message,
        "chat_history": chat_history_text
    })

    answer = result.get("output", "")

    return ChatResponse(
        answer=answer,
        response_mode=result.get("response_mode"),
        action_plan=result.get("action_plan"),
        tool_calls=result.get("tool_calls", []),
        retrieved_coach_docs=result.get("retrieved_coach_docs", []),
        retrieved_shoe_docs=result.get("retrieved_shoe_docs", []),
        constraints=result.get("constraints", {}),
        note="ถ้า debug fields ว่าง แปลว่า agent ปัจจุบันยัง return เฉพาะ final answer แต่ log ยังแสดงใน terminal ได้"
    )
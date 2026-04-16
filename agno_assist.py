import asyncio
from agno.agent import Agent
from agno.team import Team,TeamMode
from agno.models.openai import OpenAILike
from agno.os import AgentOS
from agno.tools.mcp import MCPTools
from agno.db.sqlite import AsyncSqliteDb
from agno.tools.hackernews import HackerNewsTools
from agno.tools.reasoning import ReasoningTools
from pydantic import BaseModel
from fastapi import FastAPI
from starlette.responses import StreamingResponse
from textwrap import dedent



db = AsyncSqliteDb(db_file="agno.db")
model = OpenAILike(
    id="glm-4.7-flash",
    api_key="sk-23b983e9ff4b4aabb646f1380bea2f93.FCAYtWaD0wJzGtqS",
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    max_tokens=256,
    temperature=0.7, #更有人味,地温度
    top_p=0.8, #>0.9会乱说
    timeout=10, 
)

team = Team(
    name="技术助手",
    model=model,
    members=[
        Agent(
            name="搜索助手",
            model=model,
            db=db, 
            role="你负责搜索技术问题的答案",
            tools=[HackerNewsTools()],
            add_datetime_to_context=True,
            add_history_to_context=True,
            num_history_runs=3,
            update_memory_on_run=False,
            markdown=True,
            stream=True,
        ),
        Agent(
            name="回答助手",
            model=model,
            db=db,  
            role="你负责整理答案，简洁、准确、用中文回答",
            #推理
            tools=[ReasoningTools(
                add_instructions=True,
                add_few_shot=True,
            )],
            instructions=dedent("""\
                你是一个非常聪明、温柔、有耐心的思考型助手。
                语气自然、像朋友一样，不要机器腔
            """),
            add_datetime_to_context=True,
            add_history_to_context=True,
            num_history_runs=3,
            update_memory_on_run=False,
            markdown=True,
            stream=True,
        ),
    ],
    instructions=[
        "先搜索，再回答",
        "回答必须简短",
        "只使用中文",
        "不要编造信息",
    ],
    stream=True,
    mode=TeamMode.coordinate,
    markdown=True,
    update_memory_on_run=False,
    show_members_responses=True,
)

# Serve via AgentOS → streaming, auth, session isolation, API endpoints
agent_os = AgentOS(
    agents=[team.members[0], team.members[1]],
    db=db,
    tracing=False,
)
app = agent_os.get_app()

class ChatRequest(BaseModel):
    message: str
    user_id: str = "user123"

@app.post("/chat")
async def chat(request: ChatRequest):
    return await agent_os.chat(request.message, request.user_id)

@app.get("/")
async def index():
    return {"message": "Hello, World!"}

@app.post("/stream/chat")
async def stream_chat(request: ChatRequest):
    async def generate():
        async for chunk in agent_os.stream_chat(request.message, request.user_id):
            yield f"data: {chunk}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")


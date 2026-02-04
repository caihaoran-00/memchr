"""
FastAPI接口：提供HTTP API服务
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

from memory_core.manager import MemoryManager
from memory_core.models import MessageRole
from config import MemoryConfig, ConfigPresets


# 创建FastAPI应用
app = FastAPI(
    title="玩具记忆系统API",
    description="为智能对话玩具提供记忆能力的独立服务",
    version="1.0.0"
)

# 全局记忆管理器
_manager: Optional[MemoryManager] = None


def get_manager() -> MemoryManager:
    """获取记忆管理器实例"""
    global _manager
    if _manager is None:
        config = MemoryConfig()
        _manager = MemoryManager(config)
    return _manager


# ========== 请求/响应模型 ==========

class StartSessionRequest(BaseModel):
    user_id: str = Field(..., description="用户ID")
    session_id: Optional[str] = Field(None, description="会话ID，不提供则自动生成")


class StartSessionResponse(BaseModel):
    session_id: str
    user_id: str
    message: str


class AddMessageRequest(BaseModel):
    session_id: str = Field(..., description="会话ID")
    role: str = Field(..., description="消息角色: user/assistant")
    content: str = Field(..., description="消息内容")
    metadata: Optional[Dict[str, Any]] = Field(None, description="附加元数据")


class EndSessionRequest(BaseModel):
    session_id: str = Field(..., description="会话ID")
    extract_memory: bool = Field(True, description="是否提取记忆")


class EndSessionResponse(BaseModel):
    success: bool
    episode_id: Optional[str] = None
    summary: Optional[str] = None


class GetContextRequest(BaseModel):
    session_id: str = Field(..., description="会话ID")
    query: Optional[str] = Field(None, description="查询关键词")


class MemoryContextResponse(BaseModel):
    system_prompt: str = Field(..., description="记忆增强的系统提示词")
    user_name: Optional[str] = None
    user_tags: List[str] = []
    relevant_memories: List[str] = []
    known_facts: List[str] = []


class UserProfileResponse(BaseModel):
    user_id: str
    name: str = ""
    age: Optional[int] = None
    gender: str = ""
    tags: List[str] = []
    created_at: str
    updated_at: str


class UpdateProfileRequest(BaseModel):
    user_id: str
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    tags: Optional[List[str]] = None


class StatsResponse(BaseModel):
    user_id: str
    episode_count: int
    fact_count: int
    has_profile: bool


# ========== API端点 ==========

@app.get("/")
async def root():
    """健康检查"""
    return {"status": "ok", "service": "toy-memory-system", "version": "1.0.0"}


@app.post("/session/start", response_model=StartSessionResponse)
async def start_session(request: StartSessionRequest, manager: MemoryManager = Depends(get_manager)):
    """开始新会话"""
    try:
        working_memory = manager.start_session(request.user_id, request.session_id)
        return StartSessionResponse(
            session_id=working_memory.session_id,
            user_id=working_memory.user_id,
            message="会话已创建"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/message")
async def add_message(request: AddMessageRequest, manager: MemoryManager = Depends(get_manager)):
    """添加消息到会话"""
    try:
        if request.role not in ["user", "assistant", "system"]:
            raise HTTPException(status_code=400, detail="无效的消息角色")
        
        manager.add_message(
            request.session_id,
            request.role,
            request.content,
            request.metadata
        )
        return {"success": True, "message": "消息已添加"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/end", response_model=EndSessionResponse)
async def end_session(request: EndSessionRequest, manager: MemoryManager = Depends(get_manager)):
    """结束会话并提取记忆"""
    try:
        episode = await manager.end_session(request.session_id, request.extract_memory)
        return EndSessionResponse(
            success=True,
            episode_id=episode.id if episode else None,
            summary=episode.summary if episode else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/context", response_model=MemoryContextResponse)
async def get_memory_context(request: GetContextRequest, manager: MemoryManager = Depends(get_manager)):
    """获取记忆上下文（用于增强LLM对话）"""
    try:
        context = manager.get_memory_context(request.session_id, request.query)
        
        return MemoryContextResponse(
            system_prompt=context.to_system_prompt(),
            user_name=context.user_profile.name if context.user_profile else None,
            user_tags=context.user_profile.tags if context.user_profile else [],
            relevant_memories=[ep.summary for ep in context.relevant_episodes],
            known_facts=[f.to_natural_language() for f in context.relevant_facts]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/profile/{user_id}", response_model=UserProfileResponse)
async def get_user_profile(user_id: str, manager: MemoryManager = Depends(get_manager)):
    """获取用户画像"""
    profile = manager.get_user_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    return UserProfileResponse(
        user_id=profile.user_id,
        name=profile.name,
        age=profile.age,
        gender=profile.gender,
        tags=profile.tags,
        created_at=profile.created_at.isoformat(),
        updated_at=profile.updated_at.isoformat()
    )


@app.put("/profile")
async def update_user_profile(request: UpdateProfileRequest, manager: MemoryManager = Depends(get_manager)):
    """更新用户画像"""
    from memory_core.models import UserProfile
    
    profile = manager.get_user_profile(request.user_id)
    if not profile:
        profile = UserProfile(user_id=request.user_id)
    
    if request.name is not None:
        profile.name = request.name
    if request.age is not None:
        profile.age = request.age
    if request.gender is not None:
        profile.gender = request.gender
    if request.tags is not None:
        profile.tags = request.tags
    
    manager.update_user_profile(profile)
    return {"success": True, "message": "用户画像已更新"}


@app.get("/stats/{user_id}", response_model=StatsResponse)
async def get_user_stats(user_id: str, manager: MemoryManager = Depends(get_manager)):
    """获取用户统计信息"""
    stats = manager.get_stats(user_id)
    return StatsResponse(**stats)


@app.post("/maintenance/forget/{user_id}")
async def run_forgetting(user_id: str, manager: MemoryManager = Depends(get_manager)):
    """运行遗忘机制"""
    deleted = manager.run_forgetting(user_id)
    return {"success": True, "deleted_count": deleted}


@app.post("/maintenance/cleanup")
async def cleanup_old_data(days: int = 7, manager: MemoryManager = Depends(get_manager)):
    """清理过期数据"""
    deleted = manager.cleanup(days)
    return {"success": True, "deleted_sessions": deleted}


@app.get("/export/{user_id}")
async def export_user_memory(user_id: str, manager: MemoryManager = Depends(get_manager)):
    """导出用户记忆"""
    data = manager.export_user_memory(user_id)
    return data


@app.post("/import")
async def import_user_memory(data: Dict[str, Any], manager: MemoryManager = Depends(get_manager)):
    """导入用户记忆"""
    manager.import_user_memory(data)
    return {"success": True, "message": "记忆导入成功"}


# ========== 启动入口 ==========

def create_app(config: MemoryConfig = None) -> FastAPI:
    """创建应用实例"""
    global _manager
    _manager = MemoryManager(config or MemoryConfig())
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

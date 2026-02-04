"""
记忆数据模型定义
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum
import uuid
import json


class MemoryType(Enum):
    """记忆类型"""
    WORKING = "working"      # 工作记忆（当前对话）
    EPISODIC = "episodic"    # 情景记忆（对话摘要/事件）
    SEMANTIC = "semantic"    # 语义记忆（用户画像/知识）


class MessageRole(Enum):
    """消息角色"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    """单条对话消息"""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Message":
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )


@dataclass
class WorkingMemory:
    """工作记忆：当前对话上下文"""
    user_id: str
    session_id: str
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_message(self, role: MessageRole, content: str, metadata: Dict = None):
        """添加消息"""
        msg = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(msg)
        self.updated_at = datetime.now()
    
    def get_recent(self, n: int) -> List[Message]:
        """获取最近n轮对话"""
        return self.messages[-n*2:] if len(self.messages) > n*2 else self.messages
    
    def to_prompt_format(self) -> List[Dict]:
        """转换为LLM prompt格式"""
        return [{"role": m.role.value, "content": m.content} for m in self.messages]
    
    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class Episode:
    """情景记忆：对话摘要或重要事件"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    summary: str = ""                        # 摘要内容
    keywords: List[str] = field(default_factory=list)  # 关键词
    emotion: str = ""                        # 情感标签
    importance: float = 0.5                  # 重要性评分 0-1
    access_count: int = 0                    # 访问次数
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    source_session_id: str = ""              # 来源会话ID
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 可选：向量表示
    embedding: Optional[List[float]] = None
    
    def update_access(self):
        """更新访问记录"""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def calculate_strength(self, decay_days: int = 30) -> float:
        """计算记忆强度（考虑时间衰减和访问频率）"""
        days_passed = (datetime.now() - self.last_accessed).days
        time_factor = max(0, 1 - days_passed / decay_days)
        access_factor = min(1, self.access_count / 10)
        return self.importance * (0.7 * time_factor + 0.3 * access_factor)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "summary": self.summary,
            "keywords": self.keywords,
            "emotion": self.emotion,
            "importance": self.importance,
            "access_count": self.access_count,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "source_session_id": self.source_session_id,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Episode":
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            summary=data["summary"],
            keywords=data.get("keywords", []),
            emotion=data.get("emotion", ""),
            importance=data.get("importance", 0.5),
            access_count=data.get("access_count", 0),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            source_session_id=data.get("source_session_id", ""),
            metadata=data.get("metadata", {})
        )


@dataclass
class UserProfile:
    """用户画像（语义记忆的一部分）"""
    user_id: str
    name: str = ""
    age: Optional[int] = None
    gender: str = ""
    tags: List[str] = field(default_factory=list)        # 兴趣/特征标签
    preferences: Dict[str, Any] = field(default_factory=dict)  # 偏好设置
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_tag(self, tag: str, max_tags: int = 20):
        """添加标签（去重，限制数量）"""
        if tag not in self.tags:
            self.tags.append(tag)
            if len(self.tags) > max_tags:
                self.tags = self.tags[-max_tags:]
            self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "tags": self.tags,
            "preferences": self.preferences,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "UserProfile":
        return cls(
            user_id=data["user_id"],
            name=data.get("name", ""),
            age=data.get("age"),
            gender=data.get("gender", ""),
            tags=data.get("tags", []),
            preferences=data.get("preferences", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now()
        )


@dataclass
class Fact:
    """知识事实（语义记忆的一部分）"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    subject: str = ""          # 主体 (如: "小明")
    predicate: str = ""        # 谓词 (如: "喜欢")
    object: str = ""           # 客体 (如: "恐龙")
    confidence: float = 1.0    # 置信度
    source: str = ""           # 来源（哪次对话提取的）
    created_at: datetime = field(default_factory=datetime.now)
    last_verified: datetime = field(default_factory=datetime.now)
    
    def to_natural_language(self) -> str:
        """转换为自然语言"""
        return f"{self.subject}{self.predicate}{self.object}"
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "last_verified": self.last_verified.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Fact":
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            subject=data.get("subject", ""),
            predicate=data.get("predicate", ""),
            object=data.get("object", ""),
            confidence=data.get("confidence", 1.0),
            source=data.get("source", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            last_verified=datetime.fromisoformat(data["last_verified"]) if "last_verified" in data else datetime.now()
        )


@dataclass  
class MemoryContext:
    """记忆上下文：整合各类记忆供LLM使用"""
    working_memory: Optional[WorkingMemory] = None
    relevant_episodes: List[Episode] = field(default_factory=list)
    user_profile: Optional[UserProfile] = None
    relevant_facts: List[Fact] = field(default_factory=list)
    
    def to_system_prompt(self) -> str:
        """生成记忆增强的系统提示词"""
        parts = []
        
        # 用户画像
        if self.user_profile:
            profile_info = []
            if self.user_profile.name:
                profile_info.append(f"用户名字：{self.user_profile.name}")
            if self.user_profile.age:
                profile_info.append(f"年龄：{self.user_profile.age}岁")
            if self.user_profile.tags:
                profile_info.append(f"兴趣特征：{', '.join(self.user_profile.tags)}")
            if profile_info:
                parts.append("【用户信息】\n" + "\n".join(profile_info))
        
        # 相关知识
        if self.relevant_facts:
            facts_text = "\n".join([f"- {f.to_natural_language()}" for f in self.relevant_facts])
            parts.append(f"【已知信息】\n{facts_text}")
        
        # 相关历史
        if self.relevant_episodes:
            episodes_text = "\n".join([f"- {e.summary}" for e in self.relevant_episodes[:3]])
            parts.append(f"【相关记忆】\n{episodes_text}")
        
        return "\n\n".join(parts) if parts else ""
    
    def to_dict(self) -> Dict:
        return {
            "working_memory": self.working_memory.to_dict() if self.working_memory else None,
            "relevant_episodes": [e.to_dict() for e in self.relevant_episodes],
            "user_profile": self.user_profile.to_dict() if self.user_profile else None,
            "relevant_facts": [f.to_dict() for f in self.relevant_facts]
        }

"""
记忆系统核心数据模型。
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class MemoryType(Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    """单条对话消息。"""

    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class WorkingMemory:
    """当前会话的工作记忆。"""

    user_id: str
    session_id: str
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_message(
        self, role: MessageRole, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        self.messages.append(
            Message(role=role, content=content, metadata=metadata or {})
        )
        self.updated_at = datetime.now()

    def get_recent(self, n: int) -> List[Message]:
        if len(self.messages) <= n * 2:
            return self.messages
        return self.messages[-n * 2 :]

    def to_prompt_format(self) -> List[Dict[str, str]]:
        return [{"role": message.role.value, "content": message.content} for message in self.messages]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "messages": [message.to_dict() for message in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkingMemory":
        return cls(
            user_id=data["user_id"],
            session_id=data["session_id"],
            messages=[Message.from_dict(item) for item in data.get("messages", [])],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


@dataclass
class Episode:
    """会话压缩后的情景记忆。"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    summary: str = ""
    keywords: List[str] = field(default_factory=list)
    emotion: str = ""
    importance: float = 0.5
    access_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    source_session_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def update_access(self) -> None:
        self.access_count += 1
        self.last_accessed = datetime.now()

    def calculate_strength(self, decay_days: int = 30) -> float:
        days_passed = max((datetime.now() - self.last_accessed).days, 0)
        time_factor = max(0.0, 1 - days_passed / decay_days)
        access_factor = min(1.0, self.access_count / 10)
        return self.importance * (0.7 * time_factor + 0.3 * access_factor)

    def to_dict(self) -> Dict[str, Any]:
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
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
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
            metadata=data.get("metadata", {}),
        )


@dataclass
class UserProfile:
    """用户画像。"""

    user_id: str
    name: str = ""
    age: Optional[int] = None
    gender: str = ""
    tags: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_tag(self, tag: str, max_tags: int = 20) -> None:
        if tag and tag not in self.tags:
            self.tags.append(tag)
            if len(self.tags) > max_tags:
                self.tags = self.tags[-max_tags:]
            self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "tags": self.tags,
            "preferences": self.preferences,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        return cls(
            user_id=data["user_id"],
            name=data.get("name", ""),
            age=data.get("age"),
            gender=data.get("gender", ""),
            tags=data.get("tags", []),
            preferences=data.get("preferences", {}),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if "updated_at" in data
            else datetime.now(),
        )


@dataclass
class Fact:
    """结构化事实。"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    subject: str = ""
    predicate: str = ""
    object: str = ""
    confidence: float = 1.0
    source: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    last_verified: datetime = field(default_factory=datetime.now)

    def to_natural_language(self) -> str:
        return f"{self.subject}{self.predicate}{self.object}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "last_verified": self.last_verified.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Fact":
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            subject=data.get("subject", ""),
            predicate=data.get("predicate", ""),
            object=data.get("object", ""),
            confidence=data.get("confidence", 1.0),
            source=data.get("source", ""),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(),
            last_verified=datetime.fromisoformat(data["last_verified"])
            if "last_verified" in data
            else datetime.now(),
        )


@dataclass
class MemoryContext:
    """上层模型使用的记忆上下文。"""

    working_memory: Optional[WorkingMemory] = None
    relevant_episodes: List[Episode] = field(default_factory=list)
    user_profile: Optional[UserProfile] = None
    relevant_facts: List[Fact] = field(default_factory=list)

    def to_system_prompt(self) -> str:
        parts: List[str] = []

        if self.user_profile:
            profile_lines: List[str] = []
            if self.user_profile.name:
                profile_lines.append(f"用户名字：{self.user_profile.name}")
            if self.user_profile.age is not None:
                profile_lines.append(f"用户年龄：{self.user_profile.age}")
            if self.user_profile.gender:
                profile_lines.append(f"用户性别：{self.user_profile.gender}")
            if self.user_profile.tags:
                profile_lines.append(f"用户标签：{', '.join(self.user_profile.tags)}")
            if self.user_profile.preferences:
                profile_lines.append(f"用户偏好：{self.user_profile.preferences}")
            if profile_lines:
                parts.append("用户画像\n" + "\n".join(profile_lines))

        if self.relevant_facts:
            fact_lines = [f"- {fact.to_natural_language()}" for fact in self.relevant_facts]
            parts.append("相关事实\n" + "\n".join(fact_lines))

        if self.relevant_episodes:
            episode_lines = [f"- {episode.summary}" for episode in self.relevant_episodes[:3]]
            parts.append("相关历史片段\n" + "\n".join(episode_lines))

        return "\n\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "working_memory": self.working_memory.to_dict()
            if self.working_memory
            else None,
            "relevant_episodes": [episode.to_dict() for episode in self.relevant_episodes],
            "user_profile": self.user_profile.to_dict() if self.user_profile else None,
            "relevant_facts": [fact.to_dict() for fact in self.relevant_facts],
        }

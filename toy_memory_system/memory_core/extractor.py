"""
记忆提取器：从对话中提取记忆信息
"""
import os
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_core.models import (
    Message, Episode, Fact, UserProfile, MessageRole
)
from memory_core.llm_client import LLMClient, create_llm_client
from config import MemoryConfig


# 记忆提取的Prompt模板
EXTRACTION_PROMPT = """请分析以下对话内容，提取关键记忆信息。

对话内容：
{conversation}

请以JSON格式返回，包含以下字段：
{{
    "summary": "对话的简短摘要（不超过100字）",
    "keywords": ["关键词列表，3-5个"],
    "emotion": "对话情感（开心/难过/生气/害怕/好奇/平静）",
    "importance": 0.5,  // 重要性评分0-1，日常闲聊0.3，重要信息0.7以上
    "facts": [  // 提取的事实信息
        {{"subject": "主语", "predicate": "谓语", "object": "宾语"}}
    ],
    "profile_updates": {{  // 用户画像更新
        "name": "用户名字（如果提到）",
        "age": null,  // 年龄（如果提到）
        "tags": ["新发现的兴趣/特征标签"]
    }}
}}

注意：
1. 只提取明确提到的信息，不要推测
2. facts中的三元组要简洁准确
3. 儿童对话特别关注：喜好、害怕的事物、家庭成员、学校生活
"""


class MemoryExtractor:
    """记忆提取器：从对话中提取结构化记忆"""
    
    def __init__(self, config: MemoryConfig, llm_client: LLMClient = None):
        self.config = config
        self.llm = llm_client or create_llm_client(config)
    
    async def extract_from_conversation(
        self,
        messages: List[Message],
        user_id: str,
        session_id: str = ""
    ) -> Dict:
        """从对话中提取记忆信息"""
        
        # 格式化对话内容
        conversation_text = self._format_conversation(messages)
        
        # 调用LLM提取
        prompt = EXTRACTION_PROMPT.format(conversation=conversation_text)
        
        schema_hint = """
期望的JSON结构：
- summary: 字符串
- keywords: 字符串数组
- emotion: 字符串
- importance: 0-1的浮点数
- facts: 对象数组，每个对象有subject/predicate/object
- profile_updates: 对象，包含name/age/tags
"""
        
        result = await self.llm.extract_json(prompt, schema_hint)
        
        # 验证和清理结果
        return self._validate_extraction(result, user_id, session_id)
    
    def _format_conversation(self, messages: List[Message]) -> str:
        """格式化对话为文本"""
        lines = []
        for msg in messages:
            role = "用户" if msg.role == MessageRole.USER else "助手"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)
    
    def _validate_extraction(
        self, 
        result: Dict, 
        user_id: str,
        session_id: str
    ) -> Dict:
        """验证和清理提取结果"""
        validated = {
            "summary": result.get("summary", "")[:self.config.episode_summary_max_length],
            "keywords": result.get("keywords", [])[:5],
            "emotion": result.get("emotion", "平静"),
            "importance": min(1.0, max(0.0, float(result.get("importance", 0.5)))),
            "facts": [],
            "profile_updates": {},
            "user_id": user_id,
            "session_id": session_id
        }
        
        # 验证facts
        for fact in result.get("facts", [])[:10]:
            if isinstance(fact, dict) and all(k in fact for k in ["subject", "predicate", "object"]):
                validated["facts"].append({
                    "subject": str(fact["subject"])[:50],
                    "predicate": str(fact["predicate"])[:30],
                    "object": str(fact["object"])[:50]
                })
        
        # 验证profile_updates
        profile = result.get("profile_updates", {})
        if isinstance(profile, dict):
            if profile.get("name"):
                validated["profile_updates"]["name"] = str(profile["name"])[:20]
            if profile.get("age") is not None:
                try:
                    age = int(profile["age"])
                    if 0 < age < 150:
                        validated["profile_updates"]["age"] = age
                except (ValueError, TypeError):
                    pass
            if profile.get("tags"):
                validated["profile_updates"]["tags"] = [
                    str(t)[:20] for t in profile["tags"][:5]
                ]
        
        return validated
    
    def create_episode_from_extraction(
        self,
        extraction: Dict,
        user_id: str,
        session_id: str
    ) -> Episode:
        """从提取结果创建情景记忆"""
        return Episode(
            user_id=user_id,
            summary=extraction["summary"],
            keywords=extraction["keywords"],
            emotion=extraction["emotion"],
            importance=extraction["importance"],
            source_session_id=session_id,
            metadata={"extraction_time": datetime.now().isoformat()}
        )
    
    def create_facts_from_extraction(
        self,
        extraction: Dict,
        user_id: str,
        session_id: str
    ) -> List[Fact]:
        """从提取结果创建知识事实"""
        facts = []
        for fact_data in extraction.get("facts", []):
            fact = Fact(
                user_id=user_id,
                subject=fact_data["subject"],
                predicate=fact_data["predicate"],
                object=fact_data["object"],
                source=session_id
            )
            facts.append(fact)
        return facts
    
    def update_profile_from_extraction(
        self,
        profile: UserProfile,
        extraction: Dict
    ) -> UserProfile:
        """根据提取结果更新用户画像"""
        updates = extraction.get("profile_updates", {})
        
        if updates.get("name"):
            profile.name = updates["name"]
        
        if updates.get("age") is not None:
            profile.age = updates["age"]
        
        for tag in updates.get("tags", []):
            profile.add_tag(tag, self.config.max_profile_tags)
        
        profile.updated_at = datetime.now()
        return profile


class RuleBasedExtractor:
    """基于规则的提取器：不调用LLM，成本为零"""
    
    # 情感关键词映射
    EMOTION_KEYWORDS = {
        "开心": ["开心", "高兴", "快乐", "好玩", "哈哈", "太好了", "喜欢", "爱"],
        "难过": ["难过", "伤心", "哭", "不开心", "不想", "讨厌"],
        "生气": ["生气", "气死", "烦", "讨厌", "不要"],
        "害怕": ["害怕", "怕", "吓", "可怕", "恐怖"],
        "好奇": ["为什么", "怎么", "是什么", "什么是", "?", "？"],
    }
    
    # 事实提取模式
    FACT_PATTERNS = [
        (r"我叫(.+)", "我", "名字是", None),
        (r"我(.+)岁", "我", "年龄是", None),
        (r"我喜欢(.+)", "我", "喜欢", None),
        (r"我不喜欢(.+)", "我", "不喜欢", None),
        (r"我讨厌(.+)", "我", "讨厌", None),
        (r"我想(.+)", "我", "想要", None),
        (r"我有(.+)", "我", "拥有", None),
        (r"我的(.+)是(.+)", "我的", None, None),  # 特殊处理
        (r"(.+)是我的(.+)", None, "是我的", None),  # 特殊处理
    ]
    
    def __init__(self, config: MemoryConfig):
        self.config = config
    
    def extract_from_conversation(
        self,
        messages: List[Message],
        user_id: str,
        session_id: str = ""
    ) -> Dict:
        """基于规则从对话中提取信息"""
        
        # 合并所有用户消息
        user_messages = [m.content for m in messages if m.role == MessageRole.USER]
        full_text = " ".join(user_messages)
        
        result = {
            "summary": self._generate_summary(messages),
            "keywords": self._extract_keywords(full_text),
            "emotion": self._detect_emotion(full_text),
            "importance": self._calculate_importance(messages),
            "facts": self._extract_facts(full_text),
            "profile_updates": self._extract_profile(full_text),
            "user_id": user_id,
            "session_id": session_id
        }
        
        return result
    
    def _generate_summary(self, messages: List[Message]) -> str:
        """生成简单摘要"""
        # 取第一条用户消息和最后一条用户消息
        user_msgs = [m for m in messages if m.role == MessageRole.USER]
        if not user_msgs:
            return ""
        
        if len(user_msgs) == 1:
            return user_msgs[0].content[:100]
        
        first = user_msgs[0].content[:40]
        last = user_msgs[-1].content[:40]
        return f"{first}...{last}"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简单的分词和频率统计
        import jieba
        words = jieba.cut(text)
        
        # 停用词
        stopwords = {"的", "了", "是", "我", "你", "吗", "啊", "呢", "吧", "嘛", "哦", "呀"}
        
        # 统计词频
        word_count = {}
        for word in words:
            word = word.strip()
            if len(word) >= 2 and word not in stopwords:
                word_count[word] = word_count.get(word, 0) + 1
        
        # 返回频率最高的关键词
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        return [w for w, c in sorted_words[:5]]
    
    def _detect_emotion(self, text: str) -> str:
        """检测情感"""
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    return emotion
        return "平静"
    
    def _calculate_importance(self, messages: List[Message]) -> float:
        """计算重要性"""
        # 基于对话长度和关键信息
        base_importance = 0.3
        
        # 对话轮数越多，可能越重要
        if len(messages) > 6:
            base_importance += 0.2
        
        # 检查是否包含重要信息
        full_text = " ".join([m.content for m in messages])
        important_keywords = ["名字", "生日", "喜欢", "害怕", "家", "学校", "朋友", "秘密"]
        for kw in important_keywords:
            if kw in full_text:
                base_importance += 0.1
        
        return min(1.0, base_importance)
    
    def _extract_facts(self, text: str) -> List[Dict]:
        """提取事实"""
        facts = []
        
        for pattern, subject, predicate, _ in self.FACT_PATTERNS:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    # 处理多个捕获组
                    if len(match) >= 2:
                        facts.append({
                            "subject": subject or match[0],
                            "predicate": predicate or "",
                            "object": match[-1]
                        })
                else:
                    facts.append({
                        "subject": subject or "我",
                        "predicate": predicate or "",
                        "object": match
                    })
        
        return facts[:10]  # 限制数量
    
    def _extract_profile(self, text: str) -> Dict:
        """提取用户画像信息"""
        profile = {}
        
        # 提取名字
        name_match = re.search(r"我叫(.{1,10}?)(?:[，。,\s]|$)", text)
        if name_match:
            profile["name"] = name_match.group(1).strip()
        
        # 提取年龄
        age_match = re.search(r"我(\d{1,2})岁", text)
        if age_match:
            profile["age"] = int(age_match.group(1))
        
        # 提取兴趣标签
        tags = []
        like_matches = re.findall(r"我喜欢(.{1,10}?)(?:[，。,\s]|$)", text)
        for match in like_matches[:3]:
            tags.append(f"喜欢{match.strip()}")
        
        if tags:
            profile["tags"] = tags
        
        return profile


def create_extractor(
    config: MemoryConfig, 
    use_llm: bool = True,
    llm_client: LLMClient = None
):
    """创建提取器"""
    if use_llm and config.llm_provider != "mock":
        return MemoryExtractor(config, llm_client)
    else:
        return RuleBasedExtractor(config)

"""
记忆提取器。
"""

import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MemoryConfig
from memory_core.llm_client import LLMClient, create_llm_client
from memory_core.models import Episode, Fact, Message, MessageRole, UserProfile


EXTRACTION_PROMPT = """你是一个儿童对话记忆抽取助手，请从下面的对话里提取长期记忆。

对话内容：
{conversation}

请只返回 JSON，格式如下：
{{
  "summary": "100-200字以内的情景摘要",
  "keywords": ["关键词1", "关键词2", "关键词3"],
  "emotion": "开心/平静/难过/害怕/生气/兴奋",
  "importance": 0.0,
  "facts": [
    {{
      "subject": "主体",
      "predicate": "关系",
      "object": "客体",
      "confidence": 0.9
    }}
  ],
  "profile_updates": {{
    "name": "名字或空字符串",
    "age": null,
    "gender": "",
    "tags": ["稳定标签1", "稳定标签2"],
    "preferences": {{
      "favorite_topic": "偏好值"
    }}
  }}
}}

要求：
1. `facts` 尽量抽取稳定、可复用的信息，不要留空；至少优先抽取名字、年龄、喜欢、害怕、偏好、常做活动。
2. `summary` 保留这次对话里最值得记住的事件和偏好。
3. `keywords` 控制在 3-5 个。
4. `importance` 必须在 0 到 1 之间。
5. 如果无法确认，不要编造。"""


class MemoryExtractor:
    """使用 LLM 将对话抽取为 Episode、Fact 和 UserProfile。"""

    def __init__(self, config: MemoryConfig, llm_client: Optional[LLMClient] = None):
        self.config = config
        self.llm = llm_client or create_llm_client(config)

    async def extract_from_conversation(
        self, messages: List[Message], user_id: str, session_id: str = ""
    ) -> Dict:
        conversation_text = self._format_conversation(messages)
        prompt = EXTRACTION_PROMPT.format(conversation=conversation_text)
        schema_hint = """
- summary: string
- keywords: list[string]
- emotion: string
- importance: float in [0, 1]
- facts: list[{subject, predicate, object, confidence}]
- profile_updates: {name, age, gender, tags, preferences}
"""
        result = await self.llm.extract_json(prompt, schema_hint)
        validated = self._validate_extraction(result, user_id, session_id)
        validated["facts"] = self._backfill_facts(validated, messages)
        return validated

    async def create_episode_from_extraction(
        self, extraction: Dict, user_id: str, session_id: str
    ) -> Episode:
        episode = Episode(
            user_id=user_id,
            summary=extraction["summary"],
            keywords=extraction["keywords"],
            emotion=extraction["emotion"],
            importance=extraction["importance"],
            source_session_id=session_id,
            metadata={"extraction_time": datetime.now().isoformat()},
        )

        if self.config.enable_vector_search and episode.summary:
            embeddings = await self.llm.embed_texts([episode.summary])
            if embeddings:
                episode.embedding = embeddings[0]

        return episode

    def create_facts_from_extraction(
        self, extraction: Dict, user_id: str, session_id: str
    ) -> List[Fact]:
        facts: List[Fact] = []
        for fact_data in extraction.get("facts", []):
            subject, predicate, obj = self._normalize_fact_triplet(
                fact_data["subject"],
                fact_data["predicate"],
                fact_data["object"],
            )
            facts.append(
                Fact(
                    user_id=user_id,
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    confidence=fact_data.get("confidence", 1.0),
                    source=session_id,
                )
            )
        return self._dedupe_fact_models(facts)

    def update_profile_from_extraction(
        self, profile: UserProfile, extraction: Dict
    ) -> UserProfile:
        updates = extraction.get("profile_updates", {})

        if updates.get("name"):
            profile.name = updates["name"]
        if updates.get("age") is not None:
            profile.age = updates["age"]
        if updates.get("gender"):
            profile.gender = updates["gender"]
        for tag in updates.get("tags", []):
            profile.add_tag(tag, self.config.max_profile_tags)
        if updates.get("preferences"):
            profile.preferences.update(updates["preferences"])

        profile.updated_at = datetime.now()
        return profile

    def _format_conversation(self, messages: List[Message]) -> str:
        lines = []
        for message in messages:
            if message.role == MessageRole.USER:
                role = "用户"
            elif message.role == MessageRole.ASSISTANT:
                role = "助手"
            else:
                role = "系统"
            lines.append(f"{role}: {message.content}")
        return "\n".join(lines)

    def _validate_extraction(self, result: Dict, user_id: str, session_id: str) -> Dict:
        validated = {
            "summary": str(result.get("summary", ""))[
                : self.config.episode_summary_max_length
            ],
            "keywords": [],
            "emotion": str(result.get("emotion", "平静"))[:20],
            "importance": 0.5,
            "facts": [],
            "profile_updates": {"tags": [], "preferences": {}},
            "user_id": user_id,
            "session_id": session_id,
        }

        try:
            validated["importance"] = min(
                1.0, max(0.0, float(result.get("importance", 0.5)))
            )
        except (TypeError, ValueError):
            validated["importance"] = 0.5

        keywords = result.get("keywords", [])
        if isinstance(keywords, list):
            validated["keywords"] = [str(item)[:20] for item in keywords[:5] if item]

        for fact in result.get("facts", [])[:10]:
            if not isinstance(fact, dict):
                continue
            subject = str(fact.get("subject", "")).strip()[:50]
            predicate = str(fact.get("predicate", "")).strip()[:30]
            obj = str(fact.get("object", "")).strip()[:50]
            if not subject or not predicate or not obj:
                continue
            try:
                confidence = min(1.0, max(0.0, float(fact.get("confidence", 1.0))))
            except (TypeError, ValueError):
                confidence = 1.0
            validated["facts"].append(
                {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "confidence": confidence,
                }
            )

        profile = result.get("profile_updates", {})
        if isinstance(profile, dict):
            if profile.get("name"):
                validated["profile_updates"]["name"] = str(profile["name"]).strip()[:20]
            if profile.get("age") is not None:
                try:
                    age = int(profile["age"])
                    if 0 < age < 150:
                        validated["profile_updates"]["age"] = age
                except (TypeError, ValueError):
                    pass
            if profile.get("gender"):
                validated["profile_updates"]["gender"] = str(profile["gender"]).strip()[:10]
            if isinstance(profile.get("tags"), list):
                validated["profile_updates"]["tags"] = [
                    str(tag).strip()[:20] for tag in profile["tags"][:5] if tag
                ]
            if isinstance(profile.get("preferences"), dict):
                validated["profile_updates"]["preferences"] = profile["preferences"]

        return validated

    def _backfill_facts(self, extraction: Dict, messages: List[Message]) -> List[Dict]:
        existing = extraction.get("facts", [])
        dedup = {}
        for fact in existing:
            subject, predicate, obj = self._normalize_fact_triplet(
                fact["subject"], fact["predicate"], fact["object"]
            )
            dedup[(subject, predicate, obj)] = {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "confidence": fact.get("confidence", 1.0),
            }

        profile = extraction.get("profile_updates", {})
        subject = profile.get("name") or "用户"

        age = profile.get("age")
        if age is not None:
            dedup.setdefault(
                (subject, "年龄", f"{age}岁"),
                {"subject": subject, "predicate": "年龄", "object": f"{age}岁", "confidence": 0.85},
            )

        for tag in profile.get("tags", []):
            parsed = self._fact_from_tag(subject, tag)
            if parsed:
                dedup.setdefault((parsed["subject"], parsed["predicate"], parsed["object"]), parsed)

        preferences = profile.get("preferences", {})
        for key, value in preferences.items():
            text = str(value).strip()
            if not text:
                continue
            predicate = "偏好"
            if "favorite" in key.lower() or "topic" in key.lower():
                predicate = "喜欢"
            dedup.setdefault(
                (subject, predicate, text),
                {"subject": subject, "predicate": predicate, "object": text, "confidence": 0.8},
            )

        if not dedup:
            for message in messages:
                if message.role != MessageRole.USER:
                    continue
                for parsed in self._facts_from_user_text(message.content, subject):
                    dedup.setdefault(
                        (parsed["subject"], parsed["predicate"], parsed["object"]), parsed
                    )

        return list(dedup.values())[:10]

    def _dedupe_fact_models(self, facts: List[Fact]) -> List[Fact]:
        dedup: Dict[tuple[str, str, str], Fact] = {}
        for fact in facts:
            key = self._normalize_fact_triplet(fact.subject, fact.predicate, fact.object)
            existing = dedup.get(key)
            if not existing or fact.confidence > existing.confidence:
                fact.subject, fact.predicate, fact.object = key
                dedup[key] = fact
        return list(dedup.values())

    def _normalize_fact_triplet(
        self, subject: str, predicate: str, obj: str
    ) -> tuple[str, str, str]:
        subject = self._normalize_fact_text(subject)
        predicate = self._normalize_fact_text(predicate)
        obj = self._normalize_fact_text(obj)

        predicate_aliases = {
            "名字": "名字是",
            "名字是": "名字是",
            "年龄": "年龄是",
            "年龄是": "年龄是",
        }
        predicate = predicate_aliases.get(predicate, predicate)

        if predicate in {"名字是", "年龄是"} and obj.startswith("是"):
            obj = obj[1:].strip()

        return subject, predicate, obj

    def _normalize_fact_text(self, value: str) -> str:
        return " ".join(str(value).strip().split())

    def _fact_from_tag(self, subject: str, tag: str) -> Optional[Dict]:
        tag = tag.strip()
        if not tag:
            return None
        if tag.startswith("喜欢") and len(tag) > 2:
            return {
                "subject": subject,
                "predicate": "喜欢",
                "object": tag[2:],
                "confidence": 0.82,
            }
        if tag.startswith("害怕") and len(tag) > 2:
            return {
                "subject": subject,
                "predicate": "害怕",
                "object": tag[2:],
                "confidence": 0.82,
            }
        if tag.startswith("爱") and len(tag) > 1:
            return {
                "subject": subject,
                "predicate": "喜欢",
                "object": tag[1:],
                "confidence": 0.75,
            }
        return None

    def _facts_from_user_text(self, text: str, subject: str) -> List[Dict]:
        facts: List[Dict] = []

        patterns = [
            (r"我叫([^\s，。！？,!?]{1,12})", "名字", 0.9),
            (r"我今年?(\d{1,2})岁", "年龄", 0.9),
            (r"我最喜欢([^，。！？,!?]{1,20})", "喜欢", 0.85),
            (r"我喜欢([^，。！？,!?]{1,20})", "喜欢", 0.8),
            (r"我害怕([^，。！？,!?]{1,20})", "害怕", 0.8),
        ]

        for pattern, predicate, confidence in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            value = match.group(1).strip()
            if predicate == "名字":
                facts.append(
                    {
                        "subject": subject,
                        "predicate": "名字",
                        "object": value,
                        "confidence": confidence,
                    }
                )
            elif predicate == "年龄":
                facts.append(
                    {
                        "subject": subject,
                        "predicate": "年龄",
                        "object": f"{value}岁",
                        "confidence": confidence,
                    }
                )
            else:
                facts.append(
                    {
                        "subject": subject,
                        "predicate": predicate,
                        "object": value,
                        "confidence": confidence,
                    }
                )

        return facts


def create_extractor(config: MemoryConfig, llm_client: Optional[LLMClient] = None):
    return MemoryExtractor(config, llm_client)

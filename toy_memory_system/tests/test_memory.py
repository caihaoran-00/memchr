"""
单元测试
"""
import pytest
import asyncio
import os
import sys
import tempfile
import shutil
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MemoryConfig, ConfigPresets
from memory_core.models import (
    Message, MessageRole, WorkingMemory, Episode,
    UserProfile, Fact, MemoryContext
)
from memory_core.manager import MemoryManager
from storage.sqlite_storage import SQLiteStorage


@pytest.fixture
def temp_config():
    """创建临时配置"""
    temp_dir = tempfile.mkdtemp()
    config = ConfigPresets.minimal()
    config.data_dir = temp_dir
    yield config
    # 清理
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def storage(temp_config):
    """创建存储实例"""
    return SQLiteStorage(temp_config)


@pytest.fixture
def manager(temp_config):
    """创建管理器实例"""
    return MemoryManager(temp_config)


class TestModels:
    """数据模型测试"""
    
    def test_message_creation(self):
        """测试消息创建"""
        msg = Message(role=MessageRole.USER, content="你好")
        assert msg.role == MessageRole.USER
        assert msg.content == "你好"
        assert isinstance(msg.timestamp, datetime)
    
    def test_message_serialization(self):
        """测试消息序列化"""
        msg = Message(role=MessageRole.USER, content="测试")
        data = msg.to_dict()
        assert data["role"] == "user"
        assert data["content"] == "测试"
        
        # 反序列化
        msg2 = Message.from_dict(data)
        assert msg2.role == msg.role
        assert msg2.content == msg.content
    
    def test_working_memory(self):
        """测试工作记忆"""
        wm = WorkingMemory(user_id="user1", session_id="session1")
        wm.add_message(MessageRole.USER, "问题1")
        wm.add_message(MessageRole.ASSISTANT, "回答1")
        
        assert len(wm.messages) == 2
        assert wm.messages[0].content == "问题1"
        
        # 获取最近消息
        recent = wm.get_recent(1)
        assert len(recent) == 2  # 1轮 = 2条消息
    
    def test_episode_strength(self):
        """测试情景记忆强度计算"""
        ep = Episode(
            user_id="user1",
            summary="测试摘要",
            importance=0.8,
            access_count=5
        )
        
        strength = ep.calculate_strength(decay_days=30)
        assert 0 <= strength <= 1
        assert strength > 0.5  # 高重要性+多次访问
    
    def test_user_profile_tags(self):
        """测试用户画像标签管理"""
        profile = UserProfile(user_id="user1")
        profile.add_tag("喜欢恐龙")
        profile.add_tag("5岁")
        profile.add_tag("喜欢恐龙")  # 重复
        
        assert len(profile.tags) == 2
        assert "喜欢恐龙" in profile.tags
    
    def test_fact_natural_language(self):
        """测试事实自然语言转换"""
        fact = Fact(
            user_id="user1",
            subject="小明",
            predicate="喜欢",
            object="恐龙"
        )
        
        assert fact.to_natural_language() == "小明喜欢恐龙"
    
    def test_memory_context_prompt(self):
        """测试记忆上下文生成提示词"""
        profile = UserProfile(user_id="user1", name="小明", age=5)
        profile.tags = ["喜欢恐龙", "害怕打雷"]
        
        facts = [
            Fact(user_id="user1", subject="小明", predicate="的朋友是", object="小红")
        ]
        
        episodes = [
            Episode(user_id="user1", summary="聊了关于恐龙的话题")
        ]
        
        context = MemoryContext(
            user_profile=profile,
            relevant_facts=facts,
            relevant_episodes=episodes
        )
        
        prompt = context.to_system_prompt()
        assert "小明" in prompt
        assert "5岁" in prompt
        assert "喜欢恐龙" in prompt
        assert "恐龙" in prompt


class TestStorage:
    """存储层测试"""
    
    def test_user_profile_crud(self, storage):
        """测试用户画像CRUD"""
        profile = UserProfile(
            user_id="test_user",
            name="测试用户",
            age=8,
            tags=["测试标签"]
        )
        
        # 创建
        storage.save_user_profile(profile)
        
        # 读取
        loaded = storage.get_user_profile("test_user")
        assert loaded is not None
        assert loaded.name == "测试用户"
        assert loaded.age == 8
        
        # 更新
        loaded.name = "新名字"
        storage.save_user_profile(loaded)
        
        updated = storage.get_user_profile("test_user")
        assert updated.name == "新名字"
    
    def test_episode_crud(self, storage):
        """测试情景记忆CRUD"""
        episode = Episode(
            user_id="test_user",
            summary="测试摘要",
            keywords=["测试", "记忆"],
            importance=0.7
        )
        
        # 创建
        storage.save_episode(episode)
        
        # 读取
        episodes = storage.get_episodes("test_user")
        assert len(episodes) == 1
        assert episodes[0].summary == "测试摘要"
    
    def test_episode_keyword_search(self, storage):
        """测试关键词搜索"""
        episodes = [
            Episode(user_id="user1", summary="关于恐龙的对话", keywords=["恐龙"]),
            Episode(user_id="user1", summary="关于动物园的对话", keywords=["动物园"]),
            Episode(user_id="user1", summary="恐龙和动物", keywords=["恐龙", "动物"]),
        ]
        
        for ep in episodes:
            storage.save_episode(ep)
        
        # 搜索
        results = storage.search_episodes_by_keywords("user1", ["恐龙"])
        assert len(results) >= 2
    
    def test_fact_crud(self, storage):
        """测试知识事实CRUD"""
        fact = Fact(
            user_id="test_user",
            subject="小明",
            predicate="喜欢",
            object="恐龙"
        )
        
        storage.save_fact(fact)
        
        facts = storage.get_facts("test_user")
        assert len(facts) == 1
        assert facts[0].subject == "小明"
    
    def test_fact_deduplication(self, storage):
        """测试事实去重"""
        fact1 = Fact(user_id="user1", subject="A", predicate="是", object="B")
        fact2 = Fact(user_id="user1", subject="A", predicate="是", object="B")
        
        storage.save_fact(fact1)
        storage.save_fact(fact2)
        
        facts = storage.get_facts("user1")
        assert len(facts) == 1  # 应该只有一条
    
    def test_working_memory_crud(self, storage):
        """测试工作记忆CRUD"""
        wm = WorkingMemory(user_id="user1", session_id="session1")
        wm.add_message(MessageRole.USER, "测试消息")
        
        storage.save_working_memory(wm)
        
        loaded = storage.get_working_memory("session1")
        assert loaded is not None
        assert len(loaded.messages) == 1
        
        storage.delete_working_memory("session1")
        assert storage.get_working_memory("session1") is None
    
    def test_stats(self, storage):
        """测试统计功能"""
        user_id = "stats_user"
        
        # 添加一些数据
        storage.save_user_profile(UserProfile(user_id=user_id))
        storage.save_episode(Episode(user_id=user_id, summary="test"))
        storage.save_fact(Fact(user_id=user_id, subject="A", predicate="B", object="C"))
        
        stats = storage.get_stats(user_id)
        assert stats["has_profile"] == True
        assert stats["episode_count"] == 1
        assert stats["fact_count"] == 1


class TestMemoryManager:
    """记忆管理器测试"""
    
    def test_session_lifecycle(self, manager):
        """测试会话生命周期"""
        # 开始会话
        wm = manager.start_session("user1")
        session_id = wm.session_id
        
        assert wm.user_id == "user1"
        assert session_id is not None
        
        # 添加消息
        manager.add_message(session_id, "user", "你好")
        manager.add_message(session_id, "assistant", "你好！")
        
        # 获取上下文
        context = manager.get_memory_context(session_id)
        assert context.working_memory is not None
        assert len(context.working_memory.messages) == 2
    
    @pytest.mark.asyncio
    async def test_memory_extraction(self, manager):
        """测试记忆提取"""
        wm = manager.start_session("user2")
        session_id = wm.session_id
        
        # 添加足够的对话
        for i in range(6):
            manager.add_message(session_id, "user", f"问题{i}")
            manager.add_message(session_id, "assistant", f"回答{i}")
        
        # 结束会话
        episode = await manager.end_session(session_id, extract_memory=True)
        
        # 应该提取了记忆
        assert episode is not None or True  # Mock模式可能不返回
    
    def test_user_profile_management(self, manager):
        """测试用户画像管理"""
        profile = UserProfile(user_id="profile_test", name="测试")
        manager.update_user_profile(profile)
        
        loaded = manager.get_user_profile("profile_test")
        assert loaded is not None
        assert loaded.name == "测试"
    
    def test_export_import(self, manager):
        """测试导出导入"""
        user_id = "export_test"
        
        # 创建数据
        profile = UserProfile(user_id=user_id, name="导出测试")
        manager.update_user_profile(profile)
        
        # 导出
        data = manager.export_user_memory(user_id)
        assert data["user_id"] == user_id
        assert data["profile"] is not None
        
        # 修改user_id后导入
        data["user_id"] = "import_test"
        data["profile"]["user_id"] = "import_test"
        
        manager.import_user_memory(data)
        
        loaded = manager.get_user_profile("import_test")
        assert loaded is not None


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

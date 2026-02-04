"""
演示脚本：展示记忆系统的完整功能
可以直接运行查看效果
"""
import asyncio
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MemoryConfig, ConfigPresets
from memory_core.manager import MemoryManager
from memory_core.models import MessageRole


async def demo_basic_memory():
    """基础记忆功能演示"""
    print("=" * 60)
    print("🧠 智能玩具记忆系统 - 基础功能演示")
    print("=" * 60)
    
    # 使用Mock配置（不调用真实API）
    config = ConfigPresets.minimal()
    config.data_dir = "./demo_data"
    
    manager = MemoryManager(config)
    user_id = "child_001"
    
    # 1. 开始会话
    print("\n📝 开始第一次对话...")
    session = manager.start_session(user_id)
    session_id = session.session_id
    print(f"   会话ID: {session_id}")
    
    # 2. 模拟对话
    conversations = [
        ("user", "你好！我叫小明"),
        ("assistant", "你好小明！很高兴认识你！"),
        ("user", "我今年5岁了"),
        ("assistant", "5岁的小明真可爱！你喜欢什么呢？"),
        ("user", "我喜欢恐龙，特别是霸王龙！"),
        ("assistant", "哇，霸王龙是恐龙之王！你知道霸王龙有多大吗？"),
        ("user", "我还有一个好朋友叫小红"),
        ("assistant", "小红一定也很可爱！你们经常一起玩吗？"),
    ]
    
    for role, content in conversations:
        manager.add_message(session_id, role, content)
        print(f"   {role}: {content}")
    
    # 3. 结束会话并提取记忆
    print("\n🧠 结束会话，提取记忆...")
    episode = await manager.end_session(session_id, extract_memory=True)
    
    if episode:
        print(f"   摘要: {episode.summary}")
        print(f"   关键词: {episode.keywords}")
        print(f"   情感: {episode.emotion}")
        print(f"   重要性: {episode.importance}")
    
    # 4. 查看用户画像
    print("\n👤 用户画像:")
    profile = manager.get_user_profile(user_id)
    if profile:
        print(f"   名字: {profile.name}")
        print(f"   年龄: {profile.age}")
        print(f"   标签: {profile.tags}")
    
    # 5. 查看知识事实
    print("\n📚 提取的知识事实:")
    stats = manager.get_stats(user_id)
    print(f"   情景记忆数: {stats['episode_count']}")
    print(f"   知识事实数: {stats['fact_count']}")
    
    # 6. 开始新会话并使用记忆
    print("\n📝 开始第二次对话（使用记忆）...")
    session2 = manager.start_session(user_id)
    session_id2 = session2.session_id
    
    manager.add_message(session_id2, "user", "你还记得我吗？")
    
    # 获取记忆上下文
    context = manager.get_memory_context(session_id2, "记得我")
    print("\n💡 记忆增强的系统提示词:")
    print("-" * 40)
    print(context.to_system_prompt())
    print("-" * 40)
    
    # 清理演示数据
    await manager.end_session(session_id2, extract_memory=False)
    
    print("\n✅ 演示完成！")
    print("=" * 60)
    
    return manager


async def demo_memory_retrieval():
    """记忆检索演示"""
    print("\n" + "=" * 60)
    print("🔍 记忆检索功能演示")
    print("=" * 60)
    
    config = ConfigPresets.minimal()
    config.data_dir = "./demo_data"
    
    manager = MemoryManager(config)
    user_id = "child_002"
    
    # 创建多个对话场景
    scenarios = [
        [
            ("user", "我害怕打雷"),
            ("assistant", "别怕，打雷是云朵在说话呢"),
            ("user", "真的吗？那我就不怕了"),
        ],
        [
            ("user", "今天我去动物园了"),
            ("assistant", "动物园好玩吗？你看到什么动物了？"),
            ("user", "我看到了大熊猫，它在吃竹子"),
            ("assistant", "大熊猫最喜欢吃竹子了！"),
        ],
        [
            ("user", "我的生日是6月1日"),
            ("assistant", "哇，你的生日是儿童节呢！"),
            ("user", "对呀，我生日的时候想要一个恐龙玩具"),
        ],
    ]
    
    print("\n📝 创建多个对话记忆...")
    for i, scenario in enumerate(scenarios):
        session = manager.start_session(user_id)
        for role, content in scenario:
            manager.add_message(session.session_id, role, content)
        episode = await manager.end_session(session.session_id, extract_memory=True)
        print(f"   场景{i+1}: {episode.summary if episode else '(无)'}")
    
    # 测试检索
    print("\n🔍 测试记忆检索...")
    
    # 开始新会话
    session = manager.start_session(user_id)
    
    # 用户问关于生日的问题
    manager.add_message(session.session_id, "user", "我的生日是什么时候？")
    
    context = manager.get_memory_context(session.session_id, "生日")
    print(f"\n   查询: '生日'")
    print(f"   找到 {len(context.relevant_episodes)} 条相关记忆:")
    for ep in context.relevant_episodes:
        print(f"      - {ep.summary}")
    print(f"   找到 {len(context.relevant_facts)} 条相关事实:")
    for fact in context.relevant_facts:
        print(f"      - {fact.to_natural_language()}")
    
    await manager.end_session(session.session_id, extract_memory=False)
    
    print("\n✅ 检索演示完成！")


async def demo_forgetting():
    """遗忘机制演示"""
    print("\n" + "=" * 60)
    print("🌙 遗忘机制演示")
    print("=" * 60)
    
    config = ConfigPresets.minimal()
    config.data_dir = "./demo_data"
    config.min_importance_threshold = 0.4  # 设置较高阈值以演示遗忘
    
    manager = MemoryManager(config)
    user_id = "child_003"
    
    # 创建不同重要性的记忆
    from memory_core.models import Episode
    from datetime import datetime, timedelta
    
    print("\n📝 创建不同重要性的记忆...")
    
    # 直接创建测试记忆
    episodes = [
        Episode(
            user_id=user_id,
            summary="今天天气很好",
            keywords=["天气"],
            importance=0.2,  # 低重要性
            emotion="平静"
        ),
        Episode(
            user_id=user_id,
            summary="小明说他喜欢恐龙",
            keywords=["恐龙", "喜欢"],
            importance=0.8,  # 高重要性
            emotion="开心"
        ),
        Episode(
            user_id=user_id,
            summary="随便聊了几句",
            keywords=[],
            importance=0.15,  # 极低重要性
            emotion="平静"
        ),
    ]
    
    for ep in episodes:
        manager.storage.save_episode(ep)
        print(f"   - {ep.summary} (重要性: {ep.importance})")
    
    # 查看当前记忆数量
    stats_before = manager.get_stats(user_id)
    print(f"\n📊 遗忘前: {stats_before['episode_count']} 条记忆")
    
    # 运行遗忘机制
    deleted = manager.run_forgetting(user_id)
    print(f"🗑️  删除了 {deleted} 条弱记忆")
    
    # 查看遗忘后记忆数量
    stats_after = manager.get_stats(user_id)
    print(f"📊 遗忘后: {stats_after['episode_count']} 条记忆")
    
    # 显示保留的记忆
    remaining = manager.storage.get_episodes(user_id, limit=10)
    print("\n📋 保留的记忆:")
    for ep in remaining:
        print(f"   - {ep.summary} (重要性: {ep.importance})")
    
    print("\n✅ 遗忘演示完成！")


async def demo_export_import():
    """导出/导入演示"""
    print("\n" + "=" * 60)
    print("📦 记忆导出/导入演示")
    print("=" * 60)
    
    config = ConfigPresets.minimal()
    config.data_dir = "./demo_data"
    
    manager = MemoryManager(config)
    user_id = "child_export_test"
    
    # 创建一些测试数据
    print("\n📝 创建测试数据...")
    session = manager.start_session(user_id)
    manager.add_message(session.session_id, "user", "我叫导出测试")
    manager.add_message(session.session_id, "assistant", "你好！")
    await manager.end_session(session.session_id, extract_memory=True)
    
    # 导出
    print("\n📤 导出用户记忆...")
    export_data = manager.export_user_memory(user_id)
    print(f"   用户ID: {export_data['user_id']}")
    print(f"   导出时间: {export_data['export_time']}")
    print(f"   情景记忆数: {len(export_data['episodes'])}")
    print(f"   知识事实数: {len(export_data['facts'])}")
    
    # 保存到文件
    import json
    export_path = os.path.join(config.data_dir, "memory_export.json")
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    print(f"   已保存到: {export_path}")
    
    # 模拟导入到新用户
    print("\n📥 导入到新用户...")
    export_data["user_id"] = "child_import_test"
    
    # 更新所有记忆的user_id
    if export_data.get("profile"):
        export_data["profile"]["user_id"] = "child_import_test"
    for ep in export_data.get("episodes", []):
        ep["user_id"] = "child_import_test"
    for fact in export_data.get("facts", []):
        fact["user_id"] = "child_import_test"
    
    manager.import_user_memory(export_data)
    
    # 验证导入
    new_stats = manager.get_stats("child_import_test")
    print(f"   导入后统计: {new_stats}")
    
    print("\n✅ 导出/导入演示完成！")


async def main():
    """运行所有演示"""
    print("\n" + "🎮 " * 20)
    print("     智能对话玩具记忆系统 - 完整演示")
    print("🎮 " * 20 + "\n")
    
    # 确保演示目录存在
    os.makedirs("./demo_data", exist_ok=True)
    
    try:
        await demo_basic_memory()
        await demo_memory_retrieval()
        await demo_forgetting()
        await demo_export_import()
        
        print("\n" + "=" * 60)
        print("🎉 所有演示完成！")
        print("=" * 60)
        
        print("\n📁 演示数据已保存到 ./demo_data 目录")
        print("💡 你可以查看 memory.db 文件来观察数据结构")
        print("\n🚀 下一步：")
        print("   1. 设置 LLM_API_KEY 环境变量")
        print("   2. 修改 config.py 使用真实LLM")
        print("   3. 运行 API 服务: python api/server.py")
        
    except Exception as e:
        print(f"\n❌ 演示出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

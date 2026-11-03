"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、登録されたツールを全部呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""

from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from app import tools
from app.agent import agent

BASE_TOOLS = {"search_glossary", "get_lecture_summary"}


def _parts(result, kind):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, kind)]


def test_base_tools_are_registered_and_called():
    model = TestModel()
    with agent.override(model=model):
        result = agent.run_sync("LoRAとは？")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert BASE_TOOLS <= offered  # 演習でツールを足しても壊れないよう「含む」で判定

    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert BASE_TOOLS <= called

    # 第3回 宿題（実装レベル）で save_learning_log を requires_approval=True で登録したため、
    # TestModel() が call_tools='all' で呼ぶと、それだけは承認待ちで ToolReturnPart を持たない。
    # それ以外の呼び出しは全部返っていることを確認する。
    deferred_ids = (
        {c.tool_call_id for c in result.output.approvals} | {c.tool_call_id for c in result.output.calls}
        if isinstance(result.output, DeferredToolRequests)
        else set()
    )
    resolved_calls = [p for p in _parts(result, ToolCallPart) if p.tool_call_id not in deferred_ids]
    assert len(_parts(result, ToolReturnPart)) == len(resolved_calls)
    assert result.output


def test_glossary_result_reaches_the_model():
    model = TestModel(call_tools=["search_glossary"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？")
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "search_glossary"


def test_make_quiz_tool_is_registered_and_called():
    """第3回 演習: 追加した make_quiz がエージェントに登録され、呼ばれること。"""
    model = TestModel(call_tools=["make_quiz"])
    with agent.override(model=model):
        result = agent.run_sync("第2回の内容から確認問題を1問出して")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert "make_quiz" in offered
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "make_quiz"


def test_save_learning_log_needs_approval_before_it_writes_anything(tmp_path, monkeypatch):
    """第3回 宿題（実装レベル）: 承認前はファイルが書かれず、DeferredToolRequests が返ること。"""
    log_dir = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", log_dir)

    model = TestModel(call_tools=["save_learning_log"])
    with agent.override(model=model):
        result = agent.run_sync("第3回で学んだことを記録して")

    assert isinstance(result.output, DeferredToolRequests)
    assert [c.tool_name for c in result.output.approvals] == ["save_learning_log"]
    assert not log_dir.exists()  # 承認が無いうちは tools.save_learning_log 自体が走っていない


def test_save_learning_log_writes_the_file_once_approved(tmp_path, monkeypatch):
    """第3回 宿題（実装レベル）: 承認（ToolApproved）を返すと、続きの実行で実際に保存されること。"""
    log_dir = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", log_dir)

    model = TestModel(call_tools=["save_learning_log"])
    with agent.override(model=model):
        first = agent.run_sync("第3回で学んだことを記録して")

    approvals = {c.tool_call_id: True for c in first.output.approvals}
    deferred_results = first.output.build_results(approvals=approvals)
    with agent.override(model=TestModel(call_tools=[])):
        second = agent.run_sync(
            message_history=first.all_messages(),
            deferred_tool_results=deferred_results,
        )

    assert not isinstance(second.output, DeferredToolRequests)
    assert len(list(log_dir.glob("*.md"))) == 1


def test_save_learning_log_denied_leaves_no_file(tmp_path, monkeypatch):
    """第3回 宿題（実装レベル）: 承認を拒否（ToolDenied）すると、ファイルは作られないこと。"""
    log_dir = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", log_dir)

    model = TestModel(call_tools=["save_learning_log"])
    with agent.override(model=model):
        first = agent.run_sync("第3回で学んだことを記録して")

    approvals = {c.tool_call_id: False for c in first.output.approvals}
    deferred_results = first.output.build_results(approvals=approvals)
    with agent.override(model=TestModel(call_tools=[])):
        agent.run_sync(
            message_history=first.all_messages(),
            deferred_tool_results=deferred_results,
        )

    assert not log_dir.exists()

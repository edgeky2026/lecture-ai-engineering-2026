"""第3回 宿題（発展 a）: 演習のエージェントと同じ指示・ツールを、Google の ADK で組み立て直す。

app/agent.py（Pydantic AI）との違いは「エージェントを組み立てる部分」だけで、
INSTRUCTIONS と、search_glossary・get_lecture_summary・make_quiz（演習で追加したツール）は
app/ のものをそのまま渡す。承認付きの save_learning_log（宿題・実装レベル）はここでは使わない。

動かす（リポジトリの直下で）:
    PYTHONPATH=. uv run --with google-adk adk run adk_agent/glossary_agent
"""

from google.adk.agents import Agent

from app import tools
from app.agent import INSTRUCTIONS

root_agent = Agent(
    model="gemini-3.5-flash-lite",
    name="glossary_agent",
    instruction=INSTRUCTIONS,
    tools=[tools.search_glossary, tools.get_lecture_summary, tools.make_quiz],
)

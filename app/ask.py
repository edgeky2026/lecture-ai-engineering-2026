"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

    # 第1回 宿題（実装レベル）: 会話履歴を保ったまま続けて質問する
    uv run python -m app.ask "LoRAとは何ですか？" "それはどのモデルに使えますか？"
    uv run python -m app.ask --chat     # 対話モード（標準入力から続けて質問する）

ツールを持たない「素の」呼び出し。1問だけのときは ask() のとおり会話履歴も持たない。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=question)

    # TODO(第1回 演習): response.usage_metadata から
    #   入力トークン数と出力トークン数を表示してみよう（Copilotに頼んでよい）

    return response.text


def ask_chat(questions: list[str]) -> list[str]:
    """第1回 宿題（実装）: 会話履歴を保ったまま、複数の質問に順番に答える。

    ask() は generate_content を毎回呼び直すだけで、前の質問は次の質問に渡らない。
    ここでは client.chats.create() が返す Chat オブジェクトを1つだけ作り、
    すべての質問をその同じ chat に send_message する。Chat がやり取りを履歴として
    内部に保持し、次の送信時に毎回まとめてモデルに渡すため、「それ」「さっきの」の
    ような指示語を含む続けての質問にも答えられる。
    """
    client = genai.Client(api_key=get_api_key())
    chat = client.chats.create(model=MODEL)
    return [chat.send_message(question).text for question in questions]


def run_interactive() -> None:
    """第1回 宿題（実装）: 標準入力から続けて質問を受け取る対話モード。

    ask_chat() と同じく、同じ chat に毎回 send_message することで履歴を保つ。
    空行・exit・quit・EOF（Ctrl-D）で終了する。
    """
    client = genai.Client(api_key=get_api_key())
    chat = client.chats.create(model=MODEL)
    print(f"[model] {MODEL}")
    print("対話モード：履歴を保ったまま続けて質問できます（終了: 空行 / exit / Ctrl-D）\n")
    while True:
        try:
            question = input("質問> ").strip()
        except EOFError:
            print()
            return
        if not question or question.lower() in {"exit", "quit"}:
            return
        try:
            print(chat.send_message(question).text, "\n")
        except errors.APIError as e:
            print(api_error_message(e.code, e.message), "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument(
        "questions",
        nargs="*",
        help="質問文。複数指定すると、会話履歴を保ったまま同じ順で質問する（宿題・実装レベル）",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="対話モード。標準入力から続けて質問する（宿題・実装レベル）",
    )
    args = parser.parse_args()

    if args.chat:
        run_interactive()
        return

    questions = args.questions or [DEFAULT_QUESTION]
    print(f"[model] {MODEL}")

    if len(questions) == 1:
        # 1問だけなら、従来どおり会話履歴を持たない ask() を使う
        print(f"[質問] {questions[0]}\n")
        try:
            print(ask(questions[0]))
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))
            raise SystemExit(1)
        return

    # 複数の質問は、同じ会話履歴を保ったまま順番に送る（宿題・実装レベル）
    try:
        answers = ask_chat(questions)
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        print(f"[質問{i}] {question}\n{answer}\n")


if __name__ == "__main__":
    main()

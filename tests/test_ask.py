"""第1回 宿題（実装）: 会話履歴を保つ ask_chat() の配線テスト（LLMもAPIキーも使わない）。

実際の Gemini API は呼ばず、google.genai.Client を偽物に差し替えて、
「複数の質問が、同じ Chat オブジェクト（＝同じ会話履歴）に送られるか」だけを確かめる。
"""

from unittest.mock import MagicMock, patch

from app.ask import ask_chat


def test_ask_chat_reuses_same_chat_session_across_questions(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key-for-test")

    fake_chat = MagicMock()
    fake_chat.send_message.side_effect = [
        MagicMock(text="LoRAは大規模モデルの一部だけを低ランク行列で微調整する手法です。"),
        MagicMock(text="（さっきの続き）Huらが2021年に提案しました。"),
    ]
    fake_client = MagicMock()
    fake_client.chats.create.return_value = fake_chat

    with patch("app.ask.genai.Client", return_value=fake_client) as mock_client_cls:
        answers = ask_chat(["LoRAとは何ですか？", "それは誰が提案しましたか？"])

    mock_client_cls.assert_called_once()  # Client は1回だけ作る
    fake_client.chats.create.assert_called_once()  # Chat も1回だけ作る＝履歴を共有する
    assert fake_chat.send_message.call_count == 2  # 質問ごとに同じ chat へ送る
    assert [c.args[0] for c in fake_chat.send_message.call_args_list] == [
        "LoRAとは何ですか？",
        "それは誰が提案しましたか？",
    ]
    assert answers == [
        "LoRAは大規模モデルの一部だけを低ランク行列で微調整する手法です。",
        "（さっきの続き）Huらが2021年に提案しました。",
    ]

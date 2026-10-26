# 第1回 解答

## 演習

- 講座に固有の質問（宿題の3レベル）へのモデルの回答（要約でよい）：未実施。この作業環境には Gemini API キー（`GEMINI_API_KEY`）と GitHub Codespaces が用意できず、実際の LLM 呼び出しを行っていないため。
- 用語集の正しい意味と比べて気づいたこと：（上記のため比較できず）
- 表示できたトークン数（入力 / 出力）：（上記のため未計測。`app/ask.py` の `TODO(第1回 演習)` も未着手）

## 宿題

- 取り組んだレベル：実装
- やったこと：
  - `app/ask.py` に、会話履歴を保ったまま続けて質問できる経路を追加した。
    - `ask_chat(questions: list[str]) -> list[str]`：`genai.Client` と `client.chats.create()` で作る `Chat` オブジェクトを1つだけ生成し、渡された質問をすべて同じ `chat` に `send_message()` する。既存の `ask()` は `generate_content()` を毎回呼び直すだけで前の質問が次に渡らないのに対し、`Chat` はやり取りを内部の履歴として保持し、次の送信時にまとめてモデルに渡すため、「それ」「さっきの」のような指示語を含む続けての質問に答えられるようになる。
    - `run_interactive()`：標準入力から質問を読み続け、同じ `chat` に送る対話モード（終了は空行 / `exit` / `quit` / Ctrl-D）。
    - `main()` の CLI を拡張：引数なし→既定質問1問（従来どおり）、引数1個→従来どおり `ask()`（履歴なし）、引数2個以上→`ask_chat()` で同じ履歴を共有して順番に質問、`--chat`→`run_interactive()`。既存の呼び方（`uv run python -m app.ask "質問"`）の挙動は変えていない。
  - `tests/test_ask.py` を追加し、`google.genai.Client` をモックに差し替えて配線を検証した：
    - `Client` が1回だけ作られる
    - `chats.create()` も1回だけ呼ばれる（＝複数の質問が同じ会話履歴を共有する）
    - 質問ごとに同じ `chat.send_message()` が呼ばれ、各質問に対応する返答が順番に返る
- 結果：
  - `uv run pytest -q` → 既存11件 + 新規1件の計12件すべて成功。
  - `GEMINI_API_KEY` を外した状態で `uv run python -m app.ask "質問1" "質問2"` と `uv run python -m app.ask --chat` を実行し、トレースバックではなく既存の `get_api_key()` のエラー文言（キー登録手順の案内）で止まることを確認した（想定どおりのフェイルセーフ）。
  - **未実施：実際の Gemini API を使った動作確認**（本物の `chat` が「それ」「さっきの」のような指示語を正しく解決できるかの実機確認）。この環境に `GEMINI_API_KEY` が用意できなかったため、モックによる配線検証までしか行えていない。実際に動かすには、`GEMINI_API_KEY` を環境変数または Codespaces secret に設定したうえで、例えば次のように試すとよい。
    ```bash
    uv run python -m app.ask "LoRAとは何ですか？" "それは誰が提案しましたか？"
    uv run python -m app.ask --chat
    ```
- 気づき・疑問：
  - 履歴を保つために変えたのは「毎回 `generate_content()` を呼ぶ」から「1つの `Chat` に `send_message()` を呼び続ける」への1点だけで、トークンのやり取り自体はライブラリ（`google-genai`）側が `Chat` の中で履歴を貯めて毎回まとめて送っている。自分でメッセージ配列を組み立てて毎回渡す実装ではない。
  - 会話が長くなるほど、`Chat` が毎回モデルに渡す履歴も長くなり、`README.md` に書かれている「コンテキストウィンドウ」や「無料枠のレート制限」に早く当たりやすくなるはずだが、実測はできていない（発展レベルの範囲）。
  - 本来の宿題は「自分の fork を Codespaces で開き、`GEMINI_API_KEY` を Codespaces secret に登録して」実行する前提だが、今回は個人の Google アカウントでの API キー発行や Codespaces 起動が行えない作業環境だったため、コードの実装とテストでの配線検証のみを行った。実機確認は、GEMINI_API_KEY を用意できる環境で上記コマンドを実行して追記する。

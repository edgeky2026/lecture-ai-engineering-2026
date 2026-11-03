# 第3回 解答

## この環境についての注意

この作業環境には `GEMINI_API_KEY`（実際の Gemini API キー）と GitHub Codespaces が用意できなかった。
そのため、実際の LLM は1回も呼んでいない。代わりに `pydantic_ai.models.test.TestModel`
（実際のモデルを呼ばない偽モデル）に差し替え、`app/agent.py` と `app/tools.py` の本物のコードを
そのまま動かして、ツールの登録・呼び出し・承認フローの配線を検証した（第1回の宿題と同じ考え方）。
以下はすべてその実行結果であり、架空の出力ではない。

## 演習

- 第1回（素のLLM, `app.ask`）とエージェントの回答の違い：
  第1回の `app.ask` はツールを持たないため、講座に固有の質問（`lecNN-start タグ`等）には
  一般論で答えるか「分からない」と答える（第1回 answer.md 参照）。第3回のエージェントは
  `search_glossary` で用語集を検索してから答えるよう `INSTRUCTIONS` に書かれており、
  実際に `TestModel` でも質問ごとに `search_glossary` が呼ばれて用語集の内容が
  ツール結果として返ることを確認した（下記 trace）。
- `--trace` で見えたツール呼び出し（ツール名と引数）：
  `TestModel(call_tools=["search_glossary"])` で「LoRAとQLoRAの違いは？」を実行すると、
  ```
  [tool call]   search_glossary({'query': 'a'})
  [tool return] search_glossary -> [{"term": "AIエンジニアリング", ...}]…
  [usage] RunUsage(input_tokens=139, output_tokens=69, requests=2, tool_calls=1)
  ```
  （`TestModel` は引数の内容までは模していないため `query` の値自体は固定文字列 `"a"`。
  本物の Gemini ならここに実際の検索語が入る。ツールが1回だけ呼ばれ、戻り値が
  モデルに渡っていること自体は配線として確認できた。）
- 1つの質問での LLM 呼び出し回数（`requests`）：
  上記の `RunUsage` の通り `requests=2`（1回目でツール呼び出しを決定、2回目でツール結果を見て
  最終回答を作る、の2ステップ）。`make_quiz` も同様に `requests=2` だった。
- 追加したツールと、docstring に書いた説明：
  `app/tools.py` に `make_quiz(lecture: int) -> str` を追加し、`app/agent.py` の
  `agent.tool_plain(tools.make_quiz)` で登録した。docstring は
  「指定した回（1始まりの整数）の用語集から確認問題を1問作る。『確認問題を出して』
  『第N回の内容をテストして』のように、復習用の問題を求められたときに使う。」。
  用語集 (`data/glossary.json`) からその回の用語を1つランダムに選び、
  `{"lecture": ..., "question": "「用語」の意味は？", "answer": "定義"}` という
  JSON文字列を返す。対象の回に用語が無い場合は「まだありません」と使える回の一覧を返す
  （`get_lecture_summary` と同じエラーメッセージの作り方に合わせた）。
  `TestModel(call_tools=["make_quiz"])` で「第2回の内容から確認問題を1問出して」を実行し、
  `make_quiz` が登録され・呼ばれ・結果が返ることを確認した（`tests/test_agent.py::test_make_quiz_tool_is_registered_and_called`、
  `tests/test_tools.py::test_make_quiz_returns_a_question_from_the_requested_lecture`）。

## 宿題

- 取り組んだレベル：実装
- やったこと：
  1. `app/tools.py` に `save_learning_log(lecture: int, note: str) -> str` を追加した。
     `learning_log/`（リポジトリ直下、`.gitignore` 済み）に `lecture{回数}_{タイムスタンプ}.md`
     というファイルを作り、学習メモを書き込む。
  2. `app/agent.py` で `agent.tool_plain(tools.save_learning_log, requires_approval=True)`
     として登録し、`agent` の `output_type` に `[str, DeferredToolRequests]` を追加した
     （問題文のコード例のとおり）。
  3. `main()` に `request_approvals()` を追加し、`result.output` が `DeferredToolRequests` の間、
     `approvals` に入っている呼び出し（ツール名・引数）を1件ずつ CLI で `y/N` 確認し、
     `DeferredToolResults` を組み立てて `message_history=result.all_messages()` と合わせて
     `run_sync` を呼び直す、というループにした。
  4. `TestModel` を使ったテストを `tests/test_agent.py` と `tests/test_tools.py` に追加した
     （`GEMINI_API_KEY` も実ファイル書き込みの副作用も、`monkeypatch` で隔離している）。
  5. 既存の `tests/test_agent.py::test_base_tools_are_registered_and_called` が、
     `save_learning_log` を承認待ちのまま数えて失敗するようになったため、承認待ちの
     呼び出しを除いて「戻り値の数 == 呼び出し数」を数えるように直した（気づき・疑問に詳細）。
- 結果：
  - `uv run pytest -q` → 19件すべて成功（既存12件 + 追加7件。追加分の内訳は
    `make_quiz` 2件・`save_learning_log` の保存内容1件・エージェント経由の `make_quiz` 1件・
    `save_learning_log` の承認前/承認後/拒否後の3件）。
  - `uv run python -m app.agent --list-tools` → `search_glossary` `get_lecture_summary`
    `make_quiz` `save_learning_log` の4ツールがモデルに見えており、`save_learning_log`
    の説明文に「実行前に利用者の承認が要る」ことが入っていることを確認した
    （API キー不要・実コマンドで確認済み）。
  - `TestModel` で承認フローを通し実行した結果（`tools.LEARNING_LOG_DIR` を一時フォルダに
    差し替えて実行）：
    1. 承認前：`result.output` は `DeferredToolRequests` 型。
       `approvals` に `save_learning_log({'lecture': 0, 'note': 'a'})` が1件入っており、
       学習ログフォルダはまだ**存在しない**（＝ツール本体はまだ一度も実行されていない）。
    2. 承認後（`True` を返して `run_sync(message_history=..., deferred_tool_results=...)`
       を呼び直す）：`result.output` が通常の文字列に変わり、
       `learning_log/lecture00_20261103_114655.md` が実際に作られ、内容は
       ```
       # 第0回の学習メモ

       a
       ```
       だった（`note` の値 `"a"` も `TestModel` のダミー値）。
    3. 拒否（`False`）した場合：ツール結果は `"The tool call was denied."` となり、
       ファイルは新たに作られない（ファイル数は2.の1件のまま増えない）ことを確認した。
  - 本来の確認方法（`GEMINI_API_KEY` がある環境）：
    ```bash
    uv run python -m app.agent --trace "第3回で学んだことを学習ログに保存して"
    # [承認待ち] save_learning_log(...) を実行してよいですか？ [y/N] に y と答えると
    # learning_log/ 配下にファイルが作られ、--trace に2回分の tool call/return が出るはず
    ```
- 気づき・疑問：
  - `requires_approval=True` にした時点で、そのツールは「モデルが呼ぶかどうかを決める」のと
    「実際にPythonの関数本体が実行される」のとの間に、人の承認というもう1段階が入る。
    `agent.run_sync` が一度 `DeferredToolRequests` を返して止まり、
    `message_history` と `DeferredToolResults`（`ToolApproved`/`ToolDenied` の束）を渡して
    同じ会話を再開する、という「2回の `run_sync` で1つの会話を続ける」形になっている点が、
    第1回の宿題で作った「同じ `chat` に送り続ける」履歴の持ち方とは違う（`Agent` はステートレスで、
    履歴は呼び出し側が `all_messages()` で明示的に運ぶ）ことに気づいた。
  - 既存のスキャフォールドのテスト（`test_base_tools_are_registered_and_called`）は、
    「モデルが呼んだツールの数 == 結果が返ったツールの数」を全数一致で見ていたが、
    これは「登録された全ツールが即時に実行される」という、承認待ちツールが存在しない前提の
    チェックだった。`save_learning_log` を `requires_approval=True` で `agent` に登録した時点で、
    `TestModel()`（既定で全ツールを呼ぶ）がこのツールも呼ぶため、
    「呼ばれたが（承認待ちで）まだ返っていない」呼び出しが1件増え、素朴な全数一致が崩れる。
    これはバグではなく deferred tools の設計上正しい挙動なので、テスト側を
    「承認待ちの呼び出しは数えない」ように直すのが正しい対応だと判断した。
    宿題を「実装」レベルで取り組むと、他の2レベル（基礎・発展）には出てこない
    この種の相互作用（宿題の実装が演習の既存テストの前提を変える）が起きる、というのが
    一番の発見だった。
  - 疑問：`save_learning_log` の引数 `note` は自由記述の文字列にしたが、モデルが
    何を書いて渡すかは完全にモデル任せになる。実際の Gemini で動かした場合、
    会話の要約として適切な粒度の `note` を渡してくるのか、それとも短すぎる/長すぎるかは
    `GEMINI_API_KEY` がある環境でないと確かめられず、今回は未検証のまま残っている。

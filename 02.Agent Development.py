# Databricks notebook source
# MAGIC %md
# MAGIC # AIエージェント構築ハンズオンラボ
# MAGIC
# MAGIC ## このノートブックで学ぶこと
# MAGIC
# MAGIC このハンズオンでは、Databricks上でLangChainとLangGraphを使って、
# MAGIC 実用的なAIエージェント（自律的に動作するAIアシスタント）を構築します。
# MAGIC
# MAGIC ### AIエージェントとは？
# MAGIC
# MAGIC AIエージェントは、単に質問に答えるだけでなく、以下のことができます：
# MAGIC
# MAGIC - **自分で考えて行動する**: 質問に答えるために何をすべきか判断
# MAGIC - **ツールを使う**: データベース検索やAPI呼び出しなどを実行
# MAGIC - **複数ステップの処理**: 必要に応じて複数の行動を組み合わせる
# MAGIC
# MAGIC ### 構築するエージェントの機能
# MAGIC
# MAGIC 今回作るのは「業務マニュアル検索エージェント」です：
# MAGIC
# MAGIC 1. **質問の理解**: ユーザーの質問内容を理解
# MAGIC 2. **マニュアル検索**: ベクトル検索で関連情報を取得
# MAGIC 3. **回答生成**: 検索結果を基に分かりやすく回答
# MAGIC 4. **適切な対応**: 業務外の質問には丁寧に断る
# MAGIC
# MAGIC ### ハンズオンの流れ
# MAGIC
# MAGIC 1. 環境準備（ライブラリのインストール）
# MAGIC 2. LLM（大規模言語モデル）の設定
# MAGIC 3. ベクトル検索ツールの設定
# MAGIC 4. エージェントのロジック構築
# MAGIC 5. 実際に動かして確認

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ0: 環境準備
# MAGIC
# MAGIC ### 必要なライブラリのインストール
# MAGIC
# MAGIC 以下のライブラリをインストールします：
# MAGIC
# MAGIC - **mlflow-skinny**: モデルの管理とロギング用
# MAGIC - **langgraph**: エージェントのワークフロー構築用
# MAGIC - **databricks-langchain**: DatabricksとLangChainの連携用
# MAGIC - **databricks-agents**: Databricksのエージェント機能用
# MAGIC - **uv**: 高速なパッケージインストーラー
# MAGIC
# MAGIC ### 注意事項
# MAGIC
# MAGIC インストール後、Pythonを再起動します。
# MAGIC これにより、新しくインストールしたライブラリが正しく読み込まれます。

# COMMAND ----------

# 必要なライブラリをインストール
%pip install -U -qqqq mlflow-skinny[databricks] langgraph==0.3.4 databricks-langchain databricks-agents uv

# Pythonを再起動して新しいライブラリを有効化
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ0.5: 自動ロギングの有効化
# MAGIC
# MAGIC ### MLflow自動ロギングとは
# MAGIC
# MAGIC MLflowの自動ロギング機能を有効にすると、以下の情報が自動的に記録されます：
# MAGIC
# MAGIC - **入力と出力**: エージェントへの質問と回答
# MAGIC - **使用したツール**: どのツールをいつ使ったか
# MAGIC - **処理時間**: 各ステップにかかった時間
# MAGIC - **エラー情報**: 問題が発生した場合の詳細
# MAGIC
# MAGIC ### なぜ重要なのか
# MAGIC
# MAGIC - デバッグが簡単になる
# MAGIC - パフォーマンスの分析ができる
# MAGIC - 本番環境での動作を監視できる

# COMMAND ----------

import mlflow

# LangChainの自動ロギングを有効化
mlflow.langchain.autolog()

print("✓ MLflow自動ロギングが有効になりました")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ1: LLM（大規模言語モデル）の設定
# MAGIC
# MAGIC ### LLMとは
# MAGIC
# MAGIC LLM（Large Language Model）は、人間のように文章を理解し、生成できるAIモデルです。
# MAGIC ChatGPTやClaude、Llama などが代表例です。
# MAGIC
# MAGIC ### Databricks上のLLMを使う利点
# MAGIC
# MAGIC - **セキュリティ**: データが外部に出ない
# MAGIC - **コスト管理**: 使用量を細かく管理できる
# MAGIC - **カスタマイズ**: 自社データでファインチューニング可能
# MAGIC
# MAGIC ### ChatDatabricksクラス
# MAGIC
# MAGIC LangChainの`ChatDatabricks`クラスを使うと、
# MAGIC Databricks上のLLMを簡単に呼び出せます。

# COMMAND ----------

from databricks_langchain import ChatDatabricks

# 使用するLLMのエンドポイント名
# ※環境に合わせて変更してください
LLM_ENDPOINT_NAME = "databricks-gpt-oss-20b" #"databricks-llama-4-maverick"

# LLMモデルを初期化
model = ChatDatabricks(
    endpoint=LLM_ENDPOINT_NAME
)

print(f"✓ LLMモデルを初期化しました: {LLM_ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### LLMの動作確認
# MAGIC
# MAGIC 簡単な質問でLLMが正しく動作するか確認します。

# COMMAND ----------

# テスト質問を実行
test_question = "あなたが好きな日本の大学はどこですか？"
response = model.invoke(test_question)

print(f"質問: {test_question}")
print(f"回答: {response.content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ2: ベクトル検索ツールの設定
# MAGIC
# MAGIC ### ベクトル検索とは
# MAGIC
# MAGIC ベクトル検索は、テキストの「意味」を理解して類似した情報を見つける技術です。
# MAGIC
# MAGIC **従来のキーワード検索との違い**:
# MAGIC - キーワード検索: 「休暇」で検索 → 「休暇」という単語を含む文書のみ
# MAGIC - ベクトル検索: 「休暇」で検索 → 「有給休暇」「年次休暇」「休日」なども検索
# MAGIC
# MAGIC ### VectorSearchRetrieverToolクラス
# MAGIC
# MAGIC このクラスを使うと、ベクトル検索をエージェントの「ツール」として使えます。
# MAGIC エージェントは必要に応じてこのツールを呼び出して情報を検索します。
# MAGIC
# MAGIC ### パラメータの説明
# MAGIC
# MAGIC - **index_name**: 検索対象のインデックス名（前のノートブックで作成したもの）
# MAGIC - **tool_name**: ツールの識別名（エージェントがツールを選ぶときに使用）
# MAGIC - **num_results**: 検索結果として取得する文書の数
# MAGIC - **tool_description**: ツールの説明（エージェントがいつ使うべきか判断する材料）

# COMMAND ----------

from databricks_langchain import VectorSearchRetrieverTool

# ベクトル検索インデックスの設定
# ※環境に合わせて変更してください
CATALOG_NAME = "handson"
SCHEMA_NAME = "bricks_hr"
VECTOR_SEARCH_INDEX = f"{CATALOG_NAME}.{SCHEMA_NAME}.hr_manuals_index"

# ベクトル検索ツールを作成
vector_search_tool = VectorSearchRetrieverTool(
    index_name=VECTOR_SEARCH_INDEX,
    tool_name="search_operation_manual",
    num_results=10,  # 上位10件の関連文書を取得
    tool_description="ブリックステック社の業務マニュアルを検索します。業務に関する質問に答えるために使用してください。",
    disable_notice=True  # 通知メッセージを無効化
)

print(f"✓ ベクトル検索ツールを作成しました")
print(f"  インデックス: {VECTOR_SEARCH_INDEX}")
print(f"  ツール名: search_operation_manual")

# COMMAND ----------

# MAGIC %md
# MAGIC ### ベクトル検索ツールの動作確認
# MAGIC
# MAGIC ツールが正しく動作するか、実際に検索してみます。

# COMMAND ----------

# テスト検索を実行
test_query = "このマニュアルはどういった目的で作成されていますか？"
search_results = vector_search_tool.invoke(input=test_query)

print(f"検索クエリ: {test_query}")
print(f"\n検索結果の一部:")
print(search_results[:500] + "..." if len(search_results) > 500 else search_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ2.5: ツールをモデルにバインド
# MAGIC
# MAGIC ### ツールのバインドとは
# MAGIC
# MAGIC LLMに「このツールが使えますよ」と教える作業です。
# MAGIC これにより、LLMは必要に応じてツールを呼び出すことができます。
# MAGIC
# MAGIC ### 仕組み
# MAGIC
# MAGIC 1. LLMが質問を受け取る
# MAGIC 2. 「このツールを使えば答えられそう」と判断
# MAGIC 3. ツールを呼び出す指示を出す
# MAGIC 4. エージェントがツールを実行
# MAGIC 5. 結果をLLMに渡す
# MAGIC 6. LLMが最終的な回答を生成

# COMMAND ----------

# 使用可能なツールのリスト
tools = [vector_search_tool]

# ツールをモデルにバインド
model = model.bind_tools(tools)

print(f"✓ {len(tools)} 個のツールをモデルにバインドしました")
for tool in tools:
    print(f"  - {tool.name}: {tool.description}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ3: エージェントのロジック構築
# MAGIC
# MAGIC ここからが本番です。エージェントの「脳」を作ります。
# MAGIC
# MAGIC ### エージェントの構成要素
# MAGIC
# MAGIC エージェントは以下の3つの主要な部分から構成されます：
# MAGIC
# MAGIC 1. **モデル呼び出し部分**: LLMに質問して判断を得る
# MAGIC 2. **ツール実行部分**: LLMの指示に従ってツールを実行
# MAGIC 3. **フロー制御部分**: 次に何をすべきか決定
# MAGIC
# MAGIC これらを順番に作っていきます。

# COMMAND ----------

# MAGIC %md
# MAGIC ### ステップ3-A: モデル呼び出し関数の定義
# MAGIC
# MAGIC #### システムプロンプトとは
# MAGIC
# MAGIC システムプロンプトは、エージェントの「性格」や「役割」を定義する重要な指示文です。
# MAGIC これにより、エージェントがどのように振る舞うかが決まります。
# MAGIC
# MAGIC #### 今回のシステムプロンプトの設計
# MAGIC
# MAGIC - **役割**: 業務マニュアルスペシャリスト
# MAGIC - **行動指針**:
# MAGIC   - 業務に関する質問 → マニュアルを検索して回答
# MAGIC   - 挨拶 → 丁寧に返して自己紹介
# MAGIC   - その他の質問 → 丁寧に断って業務質問を促す
# MAGIC
# MAGIC #### 前処理の役割
# MAGIC
# MAGIC システムプロンプトを会話の最初に挿入することで、
# MAGIC すべてのやり取りでエージェントの役割を維持します。

# COMMAND ----------

from langchain_core.runnables import RunnableConfig, RunnableLambda
from mlflow.langchain.chat_agent_langgraph import ChatAgentState

# システムプロンプト（エージェントの性格と行動指針）
SYSTEM_PROMPT = """あなたはブリックステック社の業務マニュアルスペシャリストです。
社内のあらゆる業務に関する質問に対して、的確かつプロフェッショナルに回答します。

【業務に関する質問への対応】
- まず業務マニュアルを検索してください
- 検索結果に基づいてのみ回答してください
- 検索結果に答えが見つからない場合：
  1. 検索クエリを変えて数回試してください
  2. それでも見つからなければ、正直に「わからない」と伝えてください

【挨拶への対応】
- 丁寧に挨拶を返してください
- 簡潔に自己紹介してください
- 業務に関する質問を促してください

【その他の質問への対応】
- 回答できない旨をはっきりと伝えてください
- 簡潔に自己紹介してください
- 業務に関する質問を促してください

【回答の心得】
- 質問者の意図をしっかりと理解してください
- わかりやすく丁寧に説明してください
- 専門用語は必要に応じて説明を加えてください"""

# システムプロンプトを会話の先頭に追加する前処理
def add_system_prompt(state):
    """
    会話履歴の先頭にシステムプロンプトを追加する関数
    
    引数:
        state: 現在の会話状態
    
    戻り値:
        システムプロンプト + 会話履歴
    """
    return [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]

# 前処理とモデルを連結
preprocessor = RunnableLambda(add_system_prompt)
model_with_system_prompt = preprocessor | model

# モデル呼び出し関数
def call_model(state: ChatAgentState, config: RunnableConfig):
    """
    LLMを呼び出して応答を取得する関数
    
    引数:
        state: 現在のエージェント状態（会話履歴など）
        config: 実行設定
    
    戻り値:
        LLMの応答を含む辞書
    """
    # モデルを実行
    response = model_with_system_prompt.invoke(state, config)
    
    # 応答をメッセージリストに追加して返す
    return {"messages": [response]}

print("✓ モデル呼び出し関数を定義しました")

# COMMAND ----------

# MAGIC %md
# MAGIC ### ステップ3-B: ツール実行関数の定義
# MAGIC
# MAGIC #### この関数の役割
# MAGIC
# MAGIC LLMが「このツールを使いたい」と指示を出したときに、
# MAGIC 実際にツールを実行する関数です。
# MAGIC
# MAGIC #### 処理の流れ
# MAGIC
# MAGIC 1. **ツール呼び出し指示の確認**: LLMの最後のメッセージを確認
# MAGIC 2. **ツールの特定**: 指定されたツール名でツールを探す
# MAGIC 3. **引数の解析**: ツールに渡す引数をJSONから解析
# MAGIC 4. **ツールの実行**: 実際にツールを呼び出す
# MAGIC 5. **結果の返却**: 実行結果をメッセージ形式で返す
# MAGIC
# MAGIC #### エラーハンドリング
# MAGIC
# MAGIC - ツールが見つからない場合
# MAGIC - 引数の解析に失敗した場合
# MAGIC - ツールの実行中にエラーが発生した場合
# MAGIC
# MAGIC これらすべてのケースで適切なエラーメッセージを返します。

# COMMAND ----------

import json

def execute_tools(state: ChatAgentState):
    """
    LLMが指示したツールを実行する関数
    
    引数:
        state: 現在のエージェント状態
    
    戻り値:
        ツール実行結果を含むメッセージのリスト
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # LLMからのツール呼び出し指示を取得
    tool_calls = last_message.get("tool_calls", [])
    
    # ツール呼び出しがない場合は何もしない
    if not tool_calls:
        return {"messages": []}
    
    print(f"ツールを実行します: {len(tool_calls)} 個のツール呼び出し")
    
    # ツール実行結果を格納するリスト
    tool_outputs = []
    
    # 各ツール呼び出しを処理
    for tool_call in tool_calls:
        # ツール情報を取得（辞書形式とオブジェクト形式の両方に対応）
        if isinstance(tool_call, dict):
            tool_name = tool_call.get("function", {}).get("name")
            tool_args = tool_call.get("function", {}).get("arguments")
            tool_id = tool_call.get("id")
        else:
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            tool_id = tool_call.id
        
        print(f"  実行中: {tool_name}")
        
        # ツールを検索して実行
        tool_result = None
        tool_found = False
        
        for tool in tools:
            if tool.name == tool_name:
                tool_found = True
                try:
                    # 引数をパース（文字列の場合はJSONとして解析）
                    if isinstance(tool_args, str):
                        args = json.loads(tool_args)
                    else:
                        args = tool_args
                    
                    # ツールを実行
                    result = tool.invoke(args)
                    tool_result = str(result)
                    print(f"    ✓ 成功: {len(tool_result)} 文字の結果を取得")
                    
                except json.JSONDecodeError as e:
                    tool_result = f"引数の解析エラー: {str(e)}"
                    print(f"    ✗ エラー: {tool_result}")
                    
                except Exception as e:
                    tool_result = f"ツール実行エラー: {str(e)}"
                    print(f"    ✗ エラー: {tool_result}")
                
                break
        
        # ツールが見つからなかった場合
        if not tool_found:
            tool_result = f"ツール '{tool_name}' が見つかりません"
            print(f"    ✗ エラー: {tool_result}")
        
        # ツール実行結果をメッセージ形式で作成
        tool_message = {
            "role": "tool",
            "content": tool_result,
            "tool_call_id": tool_id,
            "name": tool_name
        }
        tool_outputs.append(tool_message)
    
    return {"messages": tool_outputs}

print("✓ ツール実行関数を定義しました")

# COMMAND ----------

# MAGIC %md
# MAGIC ### ステップ3-C: フロー制御関数の定義
# MAGIC
# MAGIC #### この関数の役割
# MAGIC
# MAGIC エージェントの「次の行動」を決定する関数です。
# MAGIC 交通整理の役割を果たします。
# MAGIC
# MAGIC #### 判断ロジック
# MAGIC
# MAGIC LLMの最後のメッセージを確認して：
# MAGIC
# MAGIC - **ツール呼び出しがある場合** → "continue"を返す
# MAGIC   - ツール実行ノードに進む
# MAGIC   - ツール実行後、再びLLMに戻る
# MAGIC
# MAGIC - **ツール呼び出しがない場合** → "end"を返す
# MAGIC   - 最終的な回答が生成された
# MAGIC   - エージェントの処理を終了
# MAGIC
# MAGIC #### なぜこれが重要か
# MAGIC
# MAGIC この関数により、エージェントは以下のような複雑な処理を自動的に行えます：
# MAGIC
# MAGIC 1. 質問を受け取る
# MAGIC 2. 「検索が必要だ」と判断 → ツール実行
# MAGIC 3. 検索結果を受け取る
# MAGIC 4. 「もう一度検索が必要だ」と判断 → 再度ツール実行
# MAGIC 5. 十分な情報が集まった → 回答生成
# MAGIC 6. 終了

# COMMAND ----------

def should_continue(state: ChatAgentState):
    """
    次のステップを決定する関数
    
    引数:
        state: 現在のエージェント状態
    
    戻り値:
        "continue": ツール実行ノードに進む
        "end": 処理を終了
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # ツール呼び出しがあるか確認
    has_tool_calls = bool(last_message.get("tool_calls"))
    
    if has_tool_calls:
        print("→ ツール実行が必要です")
        return "continue"
    else:
        print("→ 最終回答が生成されました")
        return "end"

print("✓ フロー制御関数を定義しました")

# COMMAND ----------

# MAGIC %md
# MAGIC ### ステップ3-D: ワークフローのグラフ構築
# MAGIC
# MAGIC #### LangGraphとは
# MAGIC
# MAGIC LangGraphは、エージェントの処理フローをグラフ構造で定義するフレームワークです。
# MAGIC
# MAGIC **グラフの要素**:
# MAGIC - **ノード（Node）**: 処理を行う場所（例：LLM呼び出し、ツール実行）
# MAGIC - **エッジ（Edge）**: ノード間の移動経路
# MAGIC - **条件付きエッジ**: 条件によって次のノードが変わる経路
# MAGIC
# MAGIC #### 今回のワークフロー
# MAGIC
# MAGIC ```
# MAGIC START
# MAGIC   ↓
# MAGIC [agent] ← LLMが判断
# MAGIC   ↓
# MAGIC 条件分岐
# MAGIC   ├→ ツール呼び出しあり → [tools] → [agent]に戻る
# MAGIC   └→ ツール呼び出しなし → END
# MAGIC ```
# MAGIC
# MAGIC #### ノードの説明
# MAGIC
# MAGIC 1. **agentノード**: LLMを呼び出して次の行動を決定
# MAGIC 2. **toolsノード**: ツールを実行して結果を取得
# MAGIC
# MAGIC #### エッジの説明
# MAGIC
# MAGIC - **条件付きエッジ（agent → ?）**: should_continue関数で判断
# MAGIC - **通常のエッジ（tools → agent）**: 常にagentに戻る

# COMMAND ----------

from langgraph.graph import END, StateGraph

print("ワークフローを構築しています...")

# ステップ1: グラフの初期化
workflow = StateGraph(ChatAgentState)
print("  ✓ グラフを初期化しました")

# ステップ2: ノードの追加
workflow.add_node("agent", RunnableLambda(call_model))
print("  ✓ 'agent' ノードを追加しました（LLM呼び出し）")

workflow.add_node("tools", execute_tools)
print("  ✓ 'tools' ノードを追加しました（ツール実行）")

# ステップ3: 条件付きエッジの追加
workflow.add_conditional_edges(
    "agent",  # 開始ノード
    should_continue,  # 判断関数
    {
        "continue": "tools",  # continueの場合はtoolsノードへ
        "end": END,  # endの場合は終了
    },
)
print("  ✓ 条件付きエッジを追加しました（agent → tools/END）")

# ステップ4: 通常のエッジの追加
workflow.add_edge("tools", "agent")
print("  ✓ エッジを追加しました（tools → agent）")

# ステップ5: エントリーポイントの設定
workflow.set_entry_point("agent")
print("  ✓ エントリーポイントを設定しました（agent）")

# ステップ6: グラフのコンパイル
agent = workflow.compile()
print("  ✓ ワークフローをコンパイルしました")

print("\n✓ エージェントの構築が完了しました！")

# COMMAND ----------

# MAGIC %md
# MAGIC ### ワークフローの可視化
# MAGIC
# MAGIC 構築したエージェントのワークフローを図で確認します。
# MAGIC
# MAGIC **図の見方**:
# MAGIC - **四角形**: ノード（処理）
# MAGIC - **矢印**: エッジ（処理の流れ）
# MAGIC - **菱形**: 条件分岐
# MAGIC
# MAGIC この図を見ることで、エージェントがどのように動作するか視覚的に理解できます。

# COMMAND ----------

from IPython.display import Image, display

try:
    # エージェントのグラフ構造を可視化
    graph_image = agent.get_graph().draw_mermaid_png()
    display(Image(graph_image))
    print("✓ ワークフローの図を表示しました")
except Exception as e:
    print(f"図の表示に失敗しました: {e}")
    print("（この機能は環境によっては動作しない場合があります）")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ4: エージェントの実行
# MAGIC
# MAGIC いよいよ構築したエージェントを動かしてみます！
# MAGIC
# MAGIC ### エージェント実行の流れ
# MAGIC
# MAGIC 1. **質問を受け取る**: ユーザーからの質問
# MAGIC 2. **エージェントが処理**: 自動的に必要な処理を実行
# MAGIC    - マニュアル検索が必要なら検索
# MAGIC    - 複数回検索が必要なら繰り返す
# MAGIC 3. **回答を返す**: 最終的な回答を生成
# MAGIC
# MAGIC ### ラッパー関数の役割
# MAGIC
# MAGIC エージェントの実行を簡単にするための関数を作ります。
# MAGIC この関数を使えば、質問を渡すだけで回答が得られます。

# COMMAND ----------

def run_agent(user_question):
    """
    エージェントを実行して回答を取得する関数
    
    引数:
        user_question: ユーザーからの質問（文字列）
    
    戻り値:
        エージェントの回答（文字列）
    """
    print(f"\n{'='*60}")
    print(f"質問: {user_question}")
    print(f"{'='*60}\n")
    
    # リクエストの形式を作成
    request = {
        "messages": {
            "role": "user",
            "content": user_question
        }
    }
    
    # エージェントを実行
    print("エージェントが処理を開始します...\n")
    response = agent.invoke(request)
    
    # 最後のメッセージ（回答）を取得
    answer = response["messages"][-1]["content"]
    
    print(f"\n{'='*60}")
    print(f"回答:")
    print(f"{'='*60}")
    
    return answer

print("✓ エージェント実行関数を定義しました")

# COMMAND ----------

# MAGIC %md
# MAGIC ### テストケース1: 業務に関する質問
# MAGIC
# MAGIC エージェントの主要機能をテストします。
# MAGIC
# MAGIC **期待される動作**:
# MAGIC 1. 質問を理解
# MAGIC 2. マニュアル検索ツールを使用
# MAGIC 3. 検索結果を基に回答を生成

# COMMAND ----------

# 業務に関する質問
question_1 = "弊社の福利厚生を教えてください。"
answer_1 = run_agent(question_1)
from IPython.display import Markdown, display
display(Markdown(answer_1))

# COMMAND ----------

# MAGIC %md
# MAGIC ### テストケース2: 挨拶
# MAGIC
# MAGIC エージェントが適切に挨拶に対応できるかテストします。
# MAGIC
# MAGIC **期待される動作**:
# MAGIC 1. 丁寧に挨拶を返す
# MAGIC 2. 自己紹介をする
# MAGIC 3. 業務に関する質問を促す

# COMMAND ----------

# 挨拶
question_2 = "こんにちは！"
answer_2 = run_agent(question_2)
from IPython.display import Markdown, display
display(Markdown(answer_2))

# COMMAND ----------

# MAGIC %md
# MAGIC ### テストケース3: 業務外の質問
# MAGIC
# MAGIC エージェントが適切に範囲外の質問を断れるかテストします。
# MAGIC
# MAGIC **期待される動作**:
# MAGIC 1. 回答できない旨を丁寧に伝える
# MAGIC 2. 自己紹介をする
# MAGIC 3. 業務に関する質問を促す

# COMMAND ----------

# 業務外の質問
question_3 = "彼女と喧嘩しました。どうすれば仲直りできる？"
answer_3 = run_agent(question_3)
from IPython.display import Markdown, display
display(Markdown(answer_3))

# COMMAND ----------

# MAGIC %md
# MAGIC ### テストケース4: 複雑な業務質問（追加テスト）
# MAGIC
# MAGIC より複雑な質問で、エージェントが複数回検索を行うかテストします。

# COMMAND ----------

# 複雑な業務質問
question_4 = "有給休暇の申請プロセスを教えてください。"
answer_4 = run_agent(question_4)
from IPython.display import Markdown, display
display(Markdown(answer_4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## まとめ：ハンズオンの振り返り
# MAGIC
# MAGIC ### 学んだこと
# MAGIC
# MAGIC このハンズオンで、以下のスキルを習得しました：
# MAGIC
# MAGIC #### 1. LLMの基本的な使い方
# MAGIC - Databricks上のLLMエンドポイントへの接続
# MAGIC - ChatDatabricksクラスの使用方法
# MAGIC - システムプロンプトによる動作制御
# MAGIC
# MAGIC #### 2. ベクトル検索の統合
# MAGIC - VectorSearchRetrieverToolの設定
# MAGIC - ツールとしての検索機能の実装
# MAGIC - 検索結果の活用方法
# MAGIC
# MAGIC #### 3. エージェントの構築
# MAGIC - LangGraphによるワークフロー設計
# MAGIC - ノードとエッジの概念
# MAGIC - 条件分岐の実装
# MAGIC
# MAGIC #### 4. 実用的なエージェントの実装
# MAGIC - 複数のツールを使いこなすエージェント
# MAGIC - 適切なエラーハンドリング
# MAGIC - ユーザーフレンドリーな応答
# MAGIC
# MAGIC ### エージェントの特徴
# MAGIC
# MAGIC 構築したエージェントは以下の能力を持っています：
# MAGIC
# MAGIC ✓ **自律的な判断**: 質問に応じて適切な行動を選択  
# MAGIC ✓ **ツールの活用**: 必要に応じてマニュアル検索を実行  
# MAGIC ✓ **複数ステップの処理**: 必要なら複数回検索を実行  
# MAGIC ✓ **適切な境界設定**: 業務外の質問には丁寧に断る  
# MAGIC ✓ **ユーザーフレンドリー**: 分かりやすく丁寧な回答  
# MAGIC
# MAGIC ### 次のステップ
# MAGIC
# MAGIC このエージェントをさらに改善するアイデア：
# MAGIC
# MAGIC #### 1. ツールの追加
# MAGIC ```python
# MAGIC # 例：社内データベース検索ツール
# MAGIC database_tool = DatabaseSearchTool(
# MAGIC     tool_name="search_employee_database",
# MAGIC     tool_description="社員情報を検索します"
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC #### 2. 会話履歴の保持
# MAGIC ```python
# MAGIC # 複数ターンの会話に対応
# MAGIC conversation_history = []
# MAGIC
# MAGIC def run_agent_with_history(question):
# MAGIC     conversation_history.append({"role": "user", "content": question})
# MAGIC     response = agent.invoke({"messages": conversation_history})
# MAGIC     conversation_history.append(response["messages"][-1])
# MAGIC     return response["messages"][-1]["content"]
# MAGIC ```
# MAGIC
# MAGIC #### 3. 回答の品質向上
# MAGIC - より詳細なシステムプロンプト
# MAGIC - Few-shot例の追加
# MAGIC - 回答フォーマットの指定
# MAGIC
# MAGIC ```python
# MAGIC # 例：構造化された回答を促すプロンプト
# MAGIC IMPROVED_SYSTEM_PROMPT = """あなたはブリックステック社の業務マニュアルスペシャリストです。
# MAGIC
# MAGIC 【回答フォーマット】
# MAGIC 1. 概要：質問への簡潔な回答
# MAGIC 2. 詳細：具体的な手順や説明
# MAGIC 3. 参考情報：関連する情報や注意点
# MAGIC 4. 出典：マニュアルのページ番号
# MAGIC
# MAGIC このフォーマットに従って、分かりやすく回答してください。"""
# MAGIC ```
# MAGIC
# MAGIC #### 4. エラーハンドリングの強化
# MAGIC ```python
# MAGIC def run_agent_with_retry(question, max_retries=3):
# MAGIC     """
# MAGIC     リトライ機能付きエージェント実行
# MAGIC     """
# MAGIC     for attempt in range(max_retries):
# MAGIC         try:
# MAGIC             return run_agent(question)
# MAGIC         except Exception as e:
# MAGIC             print(f"試行 {attempt + 1}/{max_retries} 失敗: {e}")
# MAGIC             if attempt == max_retries - 1:
# MAGIC                 return "申し訳ございません。システムエラーが発生しました。"
# MAGIC             time.sleep(2 ** attempt)  # 指数バックオフ
# MAGIC ```
# MAGIC
# MAGIC #### 5. ログと分析
# MAGIC ```python
# MAGIC import time
# MAGIC from datetime import datetime
# MAGIC
# MAGIC def run_agent_with_logging(question):
# MAGIC     """
# MAGIC     ログ記録機能付きエージェント実行
# MAGIC     """
# MAGIC     start_time = time.time()
# MAGIC     timestamp = datetime.now().isoformat()
# MAGIC     
# MAGIC     try:
# MAGIC         answer = run_agent(question)
# MAGIC         duration = time.time() - start_time
# MAGIC         
# MAGIC         # ログを記録
# MAGIC         log_entry = {
# MAGIC             "timestamp": timestamp,
# MAGIC             "question": question,
# MAGIC             "answer": answer,
# MAGIC             "duration_seconds": duration,
# MAGIC             "status": "success"
# MAGIC         }
# MAGIC         
# MAGIC         # ログをテーブルに保存（例）
# MAGIC         # spark.createDataFrame([log_entry]).write.mode("append").saveAsTable("agent_logs")
# MAGIC         
# MAGIC         return answer
# MAGIC         
# MAGIC     except Exception as e:
# MAGIC         duration = time.time() - start_time
# MAGIC         log_entry = {
# MAGIC             "timestamp": timestamp,
# MAGIC             "question": question,
# MAGIC             "error": str(e),
# MAGIC             "duration_seconds": duration,
# MAGIC             "status": "error"
# MAGIC         }
# MAGIC         raise
# MAGIC ```
# MAGIC
# MAGIC #### 6. ストリーミング応答
# MAGIC ```python
# MAGIC def run_agent_streaming(question):
# MAGIC     """
# MAGIC     ストリーミング形式で回答を返す（リアルタイム表示）
# MAGIC     """
# MAGIC     request = {
# MAGIC         "messages": {
# MAGIC             "role": "user",
# MAGIC             "content": question
# MAGIC         }
# MAGIC     }
# MAGIC     
# MAGIC     # ストリーミング実行
# MAGIC     for chunk in agent.stream(request):
# MAGIC         if "messages" in chunk:
# MAGIC             for message in chunk["messages"]:
# MAGIC                 if hasattr(message, "content"):
# MAGIC                     print(message.content, end="", flush=True)
# MAGIC     print()  # 改行
# MAGIC ```
# MAGIC
# MAGIC #### 7. マルチモーダル対応
# MAGIC ```python
# MAGIC # 画像を含む質問への対応（将来的な拡張）
# MAGIC def run_agent_with_image(question, image_path):
# MAGIC     """
# MAGIC     画像を含む質問に対応
# MAGIC     """
# MAGIC     # 画像をbase64エンコード
# MAGIC     import base64
# MAGIC     with open(image_path, "rb") as img_file:
# MAGIC         img_base64 = base64.b64encode(img_file.read()).decode()
# MAGIC     
# MAGIC     request = {
# MAGIC         "messages": {
# MAGIC             "role": "user",
# MAGIC             "content": [
# MAGIC                 {"type": "text", "text": question},
# MAGIC                 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
# MAGIC             ]
# MAGIC         }
# MAGIC     }
# MAGIC     
# MAGIC     return agent.invoke(request)
# MAGIC ```
# MAGIC
# MAGIC ### 本番環境へのデプロイ
# MAGIC
# MAGIC #### ステップ1: エージェントの登録
# MAGIC ```python
# MAGIC # MLflowにエージェントを登録
# MAGIC with mlflow.start_run():
# MAGIC     mlflow.langchain.log_model(
# MAGIC         agent,
# MAGIC         artifact_path="agent",
# MAGIC         registered_model_name="hr_manual_agent"
# MAGIC     )
# MAGIC ```
# MAGIC
# MAGIC #### ステップ2: モデルサービングエンドポイントの作成
# MAGIC ```python
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
# MAGIC
# MAGIC w = WorkspaceClient()
# MAGIC
# MAGIC # エンドポイントを作成
# MAGIC w.serving_endpoints.create(
# MAGIC     name="hr-manual-agent-endpoint",
# MAGIC     config=EndpointCoreConfigInput(
# MAGIC         served_entities=[
# MAGIC             ServedEntityInput(
# MAGIC                 entity_name="hr_manual_agent",
# MAGIC                 entity_version="1",
# MAGIC                 scale_to_zero_enabled=True,
# MAGIC                 workload_size="Small"
# MAGIC             )
# MAGIC         ]
# MAGIC     )
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC #### ステップ3: エンドポイントの呼び出し
# MAGIC ```python
# MAGIC import requests
# MAGIC import os
# MAGIC
# MAGIC def call_agent_endpoint(question):
# MAGIC     """
# MAGIC     デプロイされたエージェントエンドポイントを呼び出す
# MAGIC     """
# MAGIC     endpoint_url = "https://<your-workspace>.cloud.databricks.com/serving-endpoints/hr-manual-agent-endpoint/invocations"
# MAGIC     token = os.environ.get("DATABRICKS_TOKEN")
# MAGIC     
# MAGIC     headers = {
# MAGIC         "Authorization": f"Bearer {token}",
# MAGIC         "Content-Type": "application/json"
# MAGIC     }
# MAGIC     
# MAGIC     payload = {
# MAGIC         "messages": [
# MAGIC             {"role": "user", "content": question}
# MAGIC         ]
# MAGIC     }
# MAGIC     
# MAGIC     response = requests.post(endpoint_url, json=payload, headers=headers)
# MAGIC     return response.json()
# MAGIC ```
# MAGIC
# MAGIC ### トラブルシューティング
# MAGIC
# MAGIC #### 問題1: エージェントが応答しない
# MAGIC
# MAGIC **原因**:
# MAGIC - LLMエンドポイントの問題
# MAGIC - ベクトル検索インデックスの問題
# MAGIC - ネットワークの問題
# MAGIC
# MAGIC **解決方法**:
# MAGIC ```python
# MAGIC # 各コンポーネントを個別にテスト
# MAGIC
# MAGIC # 1. LLMのテスト
# MAGIC try:
# MAGIC     test_response = model.invoke("テスト")
# MAGIC     print("✓ LLMは正常に動作しています")
# MAGIC except Exception as e:
# MAGIC     print(f"✗ LLMエラー: {e}")
# MAGIC
# MAGIC # 2. ベクトル検索のテスト
# MAGIC try:
# MAGIC     test_search = vector_search_tool.invoke({"input": "テスト"})
# MAGIC     print("✓ ベクトル検索は正常に動作しています")
# MAGIC except Exception as e:
# MAGIC     print(f"✗ ベクトル検索エラー: {e}")
# MAGIC ```
# MAGIC
# MAGIC #### 問題2: 検索結果が不正確
# MAGIC
# MAGIC **原因**:
# MAGIC - チャンク化の問題
# MAGIC - 埋め込みモデルの問題
# MAGIC - 検索クエリの問題
# MAGIC
# MAGIC **解決方法**:
# MAGIC ```python
# MAGIC # 検索結果の詳細を確認
# MAGIC def debug_search(query):
# MAGIC     results = vector_search_tool.invoke({"input": query})
# MAGIC     print(f"検索クエリ: {query}")
# MAGIC     print(f"結果の長さ: {len(results)}")
# MAGIC     print(f"結果の一部:\n{results[:500]}")
# MAGIC     return results
# MAGIC
# MAGIC debug_search("福利厚生")
# MAGIC ```
# MAGIC
# MAGIC #### 問題3: 回答の品質が低い
# MAGIC
# MAGIC **原因**:
# MAGIC - システムプロンプトが不適切
# MAGIC - 検索結果が不十分
# MAGIC - LLMの設定が不適切
# MAGIC
# MAGIC **解決方法**:
# MAGIC ```python
# MAGIC # より詳細なシステムプロンプトを使用
# MAGIC # 検索結果の数を増やす（num_results=20）
# MAGIC # 温度パラメータを調整
# MAGIC
# MAGIC model_with_params = ChatDatabricks(
# MAGIC     endpoint=LLM_ENDPOINT_NAME,
# MAGIC     temperature=0.1,  # より決定論的な応答
# MAGIC     max_tokens=1000   # より長い応答
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ### パフォーマンスの最適化
# MAGIC
# MAGIC #### 1. キャッシングの実装
# MAGIC ```python
# MAGIC from functools import lru_cache
# MAGIC
# MAGIC @lru_cache(maxsize=100)
# MAGIC def cached_search(query):
# MAGIC     """
# MAGIC     検索結果をキャッシュ
# MAGIC     """
# MAGIC     return vector_search_tool.invoke({"input": query})
# MAGIC ```
# MAGIC
# MAGIC #### 2. 並列処理
# MAGIC ```python
# MAGIC from concurrent.futures import ThreadPoolExecutor
# MAGIC
# MAGIC def process_multiple_questions(questions):
# MAGIC     """
# MAGIC     複数の質問を並列処理
# MAGIC     """
# MAGIC     with ThreadPoolExecutor(max_workers=5) as executor:
# MAGIC         results = list(executor.map(run_agent, questions))
# MAGIC     return results
# MAGIC ```
# MAGIC
# MAGIC #### 3. バッチ処理
# MAGIC ```python
# MAGIC def batch_process_questions(questions, batch_size=10):
# MAGIC     """
# MAGIC     質問をバッチで処理
# MAGIC     """
# MAGIC     results = []
# MAGIC     for i in range(0, len(questions), batch_size):
# MAGIC         batch = questions[i:i+batch_size]
# MAGIC         batch_results = process_multiple_questions(batch)
# MAGIC         results.extend(batch_results)
# MAGIC     return results
# MAGIC ```
# MAGIC
# MAGIC ### セキュリティとコンプライアンス
# MAGIC
# MAGIC #### 1. 入力のサニタイゼーション
# MAGIC ```python
# MAGIC import re
# MAGIC
# MAGIC def sanitize_input(text):
# MAGIC     """
# MAGIC     入力をサニタイズして安全性を確保
# MAGIC     """
# MAGIC     # 危険な文字列を除去
# MAGIC     text = re.sub(r'[<>]', '', text)
# MAGIC     # 長すぎる入力を制限
# MAGIC     if len(text) > 1000:
# MAGIC         text = text[:1000]
# MAGIC     return text
# MAGIC
# MAGIC def run_agent_safe(question):
# MAGIC     sanitized_question = sanitize_input(question)
# MAGIC     return run_agent(sanitized_question)
# MAGIC ```
# MAGIC
# MAGIC #### 2. アクセス制御
# MAGIC ```python
# MAGIC def run_agent_with_auth(question, user_id):
# MAGIC     """
# MAGIC     ユーザー認証付きエージェント実行
# MAGIC     """
# MAGIC     # ユーザーの権限を確認
# MAGIC     if not has_permission(user_id, "use_agent"):
# MAGIC         return "アクセスが拒否されました。"
# MAGIC     
# MAGIC     # 使用ログを記録
# MAGIC     log_usage(user_id, question)
# MAGIC     
# MAGIC     return run_agent(question)
# MAGIC ```
# MAGIC
# MAGIC #### 3. データプライバシー
# MAGIC ```python
# MAGIC def anonymize_pii(text):
# MAGIC     """
# MAGIC     個人情報を匿名化
# MAGIC     """
# MAGIC     # メールアドレスを匿名化
# MAGIC     text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
# MAGIC                   '[EMAIL]', text)
# MAGIC     # 電話番号を匿名化
# MAGIC     text = re.sub(r'\b\d{3}-\d{4}-\d{4}\b', '[PHONE]', text)
# MAGIC     return text
# MAGIC ```
# MAGIC
# MAGIC ### モニタリングとアラート
# MAGIC
# MAGIC ```python
# MAGIC def monitor_agent_performance():
# MAGIC     """
# MAGIC     エージェントのパフォーマンスを監視
# MAGIC     """
# MAGIC     # メトリクスを取得
# MAGIC     metrics = {
# MAGIC         "avg_response_time": calculate_avg_response_time(),
# MAGIC         "success_rate": calculate_success_rate(),
# MAGIC         "error_rate": calculate_error_rate(),
# MAGIC         "user_satisfaction": calculate_user_satisfaction()
# MAGIC     }
# MAGIC     
# MAGIC     # 閾値チェック
# MAGIC     if metrics["error_rate"] > 0.05:  # 5%以上
# MAGIC         send_alert("エラー率が高すぎます")
# MAGIC     
# MAGIC     if metrics["avg_response_time"] > 10:  # 10秒以上
# MAGIC         send_alert("応答時間が遅すぎます")
# MAGIC     
# MAGIC     return metrics
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 最終チェックリスト
# MAGIC
# MAGIC エージェントを本番環境にデプロイする前に、以下を確認してください：
# MAGIC
# MAGIC ### 機能面
# MAGIC - [ ] すべてのテストケースが正常に動作する
# MAGIC - [ ] エラーハンドリングが適切に実装されている
# MAGIC - [ ] 回答の品質が要件を満たしている
# MAGIC - [ ] 応答時間が許容範囲内である
# MAGIC
# MAGIC ### セキュリティ面
# MAGIC - [ ] 入力のサニタイゼーションが実装されている
# MAGIC - [ ] アクセス制御が設定されている
# MAGIC - [ ] 個人情報の保護が実装されている
# MAGIC - [ ] ログが適切に記録されている
# MAGIC
# MAGIC ### 運用面
# MAGIC - [ ] モニタリングが設定されている
# MAGIC - [ ] アラートが設定されている
# MAGIC - [ ] ドキュメントが整備されている
# MAGIC - [ ] ロールバック手順が準備されている
# MAGIC
# MAGIC ### パフォーマンス面
# MAGIC - [ ] 負荷テストが完了している
# MAGIC - [ ] キャッシングが適切に設定されている
# MAGIC - [ ] スケーリング戦略が定義されている
# MAGIC - [ ] コスト最適化が実施されている

# COMMAND ----------

# MAGIC %md
# MAGIC ## おわりに
# MAGIC
# MAGIC お疲れ様でした！このハンズオンを通じて、実用的なAIエージェントを構築する方法を学びました。

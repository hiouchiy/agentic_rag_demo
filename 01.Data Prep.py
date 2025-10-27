# Databricks notebook source
# MAGIC %md
# MAGIC # PDFファイルからテキストを抽出してベクトル検索用データを作成する
# MAGIC
# MAGIC ## このノートブックで行うこと
# MAGIC
# MAGIC このノートブックでは、PDFファイルから情報を取り出して、AIで検索できる形式に変換します。
# MAGIC 具体的には以下の5つのステップを実行します：
# MAGIC
# MAGIC 1. **PDFファイルの読み込み**  
# MAGIC    `data/` フォルダにあるPDFファイルを取得します
# MAGIC
# MAGIC 2. **テキストの抽出**  
# MAGIC    PDFの各ページから文字情報を取り出します
# MAGIC
# MAGIC 3. **チャンク化（分割）**  
# MAGIC    長い文章を適切な長さに分割します（AIが処理しやすいサイズにするため）
# MAGIC
# MAGIC 4. **メタ情報の付与**  
# MAGIC    各チャンクに「どのPDFの何ページ目か」といった情報を追加します
# MAGIC
# MAGIC 5. **Deltaテーブルへの保存**  
# MAGIC    処理したデータをDatabricksのテーブルに保存し、後でベクトル検索できるようにします
# MAGIC
# MAGIC ## なぜこの方法を使うのか
# MAGIC
# MAGIC この方法を使うと、PDFの内容を細かく制御しながら処理できます。
# MAGIC 自動処理ツールに頼らず、自分で処理の流れを管理できるため、
# MAGIC 問題が起きたときに対処しやすくなります。

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ1: 必要なツール（ライブラリ）のインストール
# MAGIC
# MAGIC Pythonで作業するために、以下のツールをインストールします：
# MAGIC
# MAGIC - **pdfplumber**: PDFからテキストを抽出するツール
# MAGIC - **pandas**: データを表形式で扱うツール
# MAGIC - **transformers**: AIモデルを使うためのツール
# MAGIC - **pysbd**: 文章を文単位に分割するツール
# MAGIC - **databricks-vectorsearch**: ベクトル検索機能を使うツール
# MAGIC
# MAGIC インストール後、Pythonを再起動して新しいツールを使えるようにします。

# COMMAND ----------

# MAGIC %pip install pdfplumber pandas databricks-vectorsearch torch transformers pysbd
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ2: 設定情報の定義
# MAGIC
# MAGIC ここでは、データを保存する場所や使用するAIモデルの名前を設定します。
# MAGIC
# MAGIC ### 各設定の意味
# MAGIC
# MAGIC - **CATALOG**: データベースの最上位の分類（例：会社名や部門名）
# MAGIC - **SCHEMA**: データベース内のグループ（例：プロジェクト名）
# MAGIC - **VOLUME**: ファイルを保存する場所の名前
# MAGIC - **EMBEDDING_MODEL_ENDPOINT**: テキストをベクトル（数値の配列）に変換するAIモデル
# MAGIC - **VECTOR_SEARCH_ENDPOINT**: ベクトル検索を実行するサービスの名前
# MAGIC - **VECTOR_INDEX_FULLNAME**: 作成するインデックス（検索用の索引）の完全な名前

# COMMAND ----------

# 設定値（環境に合わせて変更してください）
CATALOG = "handson"
SCHEMA = "bricks_hr"
VOLUME = "manuals"

# 使用するAIモデルとサービスの名前
EMBEDDING_MODEL_ENDPOINT = "databricks-gte-large-en"
VECTOR_SEARCH_ENDPOINT = "vs_endpoint"
VECTOR_INDEX_FULLNAME = f"{CATALOG}.{SCHEMA}.hr_manuals_index"

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ3: PDFからテキストを抽出する関数
# MAGIC
# MAGIC ここでは、PDFファイルを開いて、各ページの文字情報を取り出す関数を作ります。
# MAGIC
# MAGIC ### 関数の動き
# MAGIC
# MAGIC 1. PDFファイルを開く
# MAGIC 2. ページを1つずつ処理する
# MAGIC 3. 各ページからテキストを抽出する
# MAGIC 4. ページ番号とテキストをセットにして返す
# MAGIC
# MAGIC ### エラー処理
# MAGIC
# MAGIC もしページからテキストが取得できない場合は、空の文字列を返します。
# MAGIC これにより、一部のページでエラーが起きても処理を続けられます。

# COMMAND ----------

import pdfplumber
from typing import List, Tuple

def extract_pages_text(pdf_path: str) -> List[Tuple[int, str]]:
    """
    PDFファイルから各ページのテキストを抽出する関数
    
    引数:
        pdf_path: PDFファイルのパス（場所）
    
    戻り値:
        [(ページ番号, テキスト), ...] の形式のリスト
    """
    pages = []  # 結果を格納するリスト
    
    # PDFファイルを開く
    with pdfplumber.open(pdf_path) as pdf:
        # 各ページを順番に処理
        for page_number, page in enumerate(pdf.pages):
            try:
                # ページからテキストを抽出（取得できない場合は空文字）
                text = page.extract_text() or ""
            except Exception as e:
                # エラーが起きた場合は空文字を設定
                text = ""
            
            # ページ番号とテキストをセットで保存
            pages.append((page_number, text))
    
    return pages

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ4: テキストをチャンク（小さな塊）に分割する
# MAGIC
# MAGIC ### なぜチャンク化が必要なのか
# MAGIC
# MAGIC AIモデルには一度に処理できるテキストの長さに制限があります。
# MAGIC そのため、長い文章を適切な長さに分割（チャンク化）する必要があります。
# MAGIC
# MAGIC ### この関数の特徴
# MAGIC
# MAGIC 1. **正確なトークン数の計算**  
# MAGIC    実際にAIモデルが使うトークナイザー（単語分割ツール）を使って、
# MAGIC    正確な長さを測ります
# MAGIC
# MAGIC 2. **文の途中で切らない**  
# MAGIC    文の途中で分割すると意味が分からなくなるため、
# MAGIC    文の区切りを優先して分割します
# MAGIC
# MAGIC 3. **オーバーラップ（重複）の設定**  
# MAGIC    チャンク同士を少し重複させることで、
# MAGIC    文脈をまたぐ検索でも正しく情報を取得できるようにします
# MAGIC
# MAGIC 4. **短すぎる文の除外**  
# MAGIC    意味のない短い文（例：ページ番号だけ）は除外します
# MAGIC
# MAGIC ### パラメータの説明
# MAGIC
# MAGIC - **max_tokens**: 1チャンクの最大トークン数（デフォルト: 800）
# MAGIC - **overlap_tokens**: チャンク間の重複トークン数（デフォルト: 160）
# MAGIC - **min_sentence_tokens**: 処理対象とする最小文トークン数（デフォルト: 5）

# COMMAND ----------

from transformers import AutoTokenizer
import pysbd

# AIモデルと同じトークナイザーを読み込む
tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-large-en-v1.5")

def chunk_text(text: str,
               max_tokens: int = 800,
               overlap_tokens: int = 160,
               min_sentence_tokens: int = 5) -> List[str]:
    """
    テキストを適切な長さのチャンクに分割する関数
    
    引数:
        text: 分割する元のテキスト
        max_tokens: 1チャンクの最大トークン数
        overlap_tokens: チャンク間の重複トークン数
        min_sentence_tokens: 処理対象とする最小文トークン数
    
    戻り値:
        チャンク化されたテキストのリスト
    """
    
    # ステップ1: テキストを文単位に分割
    segmenter = pysbd.Segmenter(language="en", clean=False)
    sentences = segmenter.segment(text)
    
    # ステップ2: 各文のトークン数を計算
    sentence_token_ids = []
    sentence_lengths = []
    
    for sentence in sentences:
        # 文をトークンIDに変換（AIモデルが理解できる形式）
        token_ids = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_token_ids.append(token_ids)
        sentence_lengths.append(len(token_ids))
    
    # ステップ3: 文をチャンクにまとめる
    chunks = []
    current_chunk_sentences = []  # 現在のチャンクに含まれる文
    current_token_count = 0       # 現在のチャンクのトークン数
    
    for sentence, token_length in zip(sentences, sentence_lengths):
        # 短すぎる文はスキップ
        if token_length < min_sentence_tokens:
            continue
        
        # 現在のチャンクに追加できるか確認
        if current_token_count + token_length <= max_tokens:
            # 追加可能：文をチャンクに追加
            current_chunk_sentences.append(sentence)
            current_token_count += token_length
        else:
            # 追加不可：現在のチャンクを確定して新しいチャンクを開始
            if current_chunk_sentences:
                chunks.append("".join(current_chunk_sentences))
            
            # 新しいチャンクを開始
            current_chunk_sentences = [sentence]
            current_token_count = token_length
    
    # 最後のチャンクを追加
    if current_chunk_sentences:
        chunks.append("".join(current_chunk_sentences))
    
    # ステップ4: チャンク間にオーバーラップを追加
    final_chunks = []
    
    for i, chunk in enumerate(chunks):
        if i == 0:
            # 最初のチャンクはそのまま追加
            final_chunks.append(chunk)
        else:
            # 2番目以降のチャンクには前のチャンクの一部を先頭に追加
            previous_chunk = final_chunks[-1]
            
            # 前のチャンクの最後の部分を取得
            previous_tokens = tokenizer.encode(previous_chunk, add_special_tokens=False)
            overlap_token_ids = previous_tokens[-overlap_tokens:] if len(previous_tokens) >= overlap_tokens else previous_tokens
            
            # オーバーラップ部分を文字列に戻す
            overlap_text = tokenizer.decode(overlap_token_ids)
            
            # オーバーラップ + 新しいチャンクを結合
            combined_chunk = overlap_text + chunk
            final_chunks.append(combined_chunk)
    
    return final_chunks

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ5: PDFフォルダ全体を処理する関数
# MAGIC
# MAGIC ここでは、フォルダ内のすべてのPDFファイルを処理して、
# MAGIC 1つの表（DataFrame）にまとめる関数を作ります。
# MAGIC
# MAGIC ### 処理の流れ
# MAGIC
# MAGIC 1. **1つのPDFを処理する関数**
# MAGIC    - PDFからページごとにテキストを抽出
# MAGIC    - 各ページのテキストをチャンク化
# MAGIC    - チャンクごとに情報（ファイル名、ページ番号、チャンク番号、テキスト）を記録
# MAGIC
# MAGIC 2. **フォルダ全体を処理する関数**
# MAGIC    - フォルダ内のすべてのPDFファイルを見つける
# MAGIC    - 各PDFを処理
# MAGIC    - すべての結果を1つの表にまとめる

# COMMAND ----------

import os
import pandas as pd

def build_chunks_for_pdf(pdf_path: str) -> pd.DataFrame:
    """
    1つのPDFファイルを処理してチャンクのDataFrameを作成
    
    引数:
        pdf_path: PDFファイルのパス
    
    戻り値:
        チャンク情報を含むDataFrame
    """
    # PDFからページとテキストを抽出
    pages = extract_pages_text(pdf_path)
    
    # チャンク情報を格納するリスト
    records = []
    
    # 各ページを処理
    for page_number, page_text in pages:
        # 空のページはスキップ
        if not page_text.strip():
            continue
        
        # ページのテキストをチャンク化
        text_chunks = chunk_text(page_text)  # 変数名を変更
        
        # 各チャンクの情報を記録
        for chunk_index, text_content in enumerate(text_chunks):  # 変数名を変更
            records.append({
                "pdf_path": pdf_path,           # PDFファイルのパス
                "page_number": page_number,     # ページ番号
                "chunk_id": chunk_index,        # チャンク番号
                "chunk_text": text_content      # チャンクのテキスト
            })
    
    # リストをDataFrameに変換
    return pd.DataFrame(records)


def build_chunks_for_folder(pdf_folder: str) -> pd.DataFrame:
    """
    フォルダ内のすべてのPDFファイルを処理
    
    引数:
        pdf_folder: PDFファイルが入っているフォルダのパス
    
    戻り値:
        すべてのPDFのチャンク情報を含むDataFrame
    """
    all_dataframes = []
    
    # フォルダ内のファイルを1つずつ確認
    for filename in os.listdir(pdf_folder):
        # PDFファイルのみ処理
        if filename.lower().endswith(".pdf"):
            # ファイルの完全なパスを作成
            full_path = os.path.join(pdf_folder, filename)
            
            print(f"処理中: {filename}")
            
            try:
                # PDFを処理してDataFrameを取得
                df = build_chunks_for_pdf(full_path)
                all_dataframes.append(df)
                print(f"  ✓ {len(df)} 個のチャンクを作成")
            except Exception as e:
                print(f"  ✗ エラー: {e}")
                continue
    
    # PDFが1つもない場合は空のDataFrameを返す
    if not all_dataframes:
        print("警告: 処理できるPDFファイルが見つかりませんでした")
        return pd.DataFrame([], columns=["pdf_path", "page_number", "chunk_id", "chunk_text"])
    
    # すべてのDataFrameを1つに結合
    return pd.concat(all_dataframes, ignore_index=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ6: 実際にPDFを処理する
# MAGIC
# MAGIC ここで実際に `data/` フォルダ内のPDFファイルをすべて処理します。
# MAGIC
# MAGIC ### 処理内容
# MAGIC
# MAGIC 1. フォルダ内のすべてのPDFを処理
# MAGIC 2. 各チャンクに一意のID番号を付与
# MAGIC 3. 最初の10件を表示して確認

# COMMAND ----------

# PDFが保存されているフォルダのパス
pdf_folder = "./data"

# フォルダの存在確認
if not os.path.exists(pdf_folder):
    print(f"エラー: フォルダ '{pdf_folder}' が見つかりません")
    print("フォルダを作成するか、正しいパスを指定してください")
else:
    # フォルダ内のPDFファイル数を確認
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    print(f"フォルダ内のPDFファイル数: {len(pdf_files)}")
    
    if len(pdf_files) == 0:
        print("警告: PDFファイルが見つかりません")
    else:
        print(f"以下のPDFファイルを処理します:")
        for pdf_file in pdf_files:
            print(f"  - {pdf_file}")
        print()
        
        # フォルダ内のすべてのPDFを処理
        print("PDFファイルの処理を開始します...")
        chunks_dataframe = build_chunks_for_folder(pdf_folder)
        
        # 各チャンクに一意のID番号を付与
        chunks_dataframe["id"] = range(1, len(chunks_dataframe) + 1)
        
        # 処理結果のサマリーを表示
        print(f"\n{'='*50}")
        print(f"処理完了: {len(chunks_dataframe)} 個のチャンクを作成しました")
        print(f"{'='*50}\n")
        
        # データの統計情報を表示
        if not chunks_dataframe.empty:
            print("データの統計情報:")
            print(f"  - ユニークなPDFファイル数: {chunks_dataframe['pdf_path'].nunique()}")
            print(f"  - 総ページ数: {chunks_dataframe['page_number'].nunique()}")
            print(f"  - 平均チャンク文字数: {chunks_dataframe['chunk_text'].str.len().mean():.0f}")
            print(f"  - 最小チャンク文字数: {chunks_dataframe['chunk_text'].str.len().min()}")
            print(f"  - 最大チャンク文字数: {chunks_dataframe['chunk_text'].str.len().max()}")
            print()
            
            # 最初の10件を表示
            print("最初の10件のプレビュー:")
            display(chunks_dataframe.head(10))
        else:
            print("警告: チャンクが作成されませんでした")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ7: データをDeltaテーブルに保存する
# MAGIC
# MAGIC ### Deltaテーブルとは
# MAGIC
# MAGIC Deltaテーブルは、Databricksで使用する高性能なデータ保存形式です。
# MAGIC 以下の特徴があります：
# MAGIC
# MAGIC - データの変更履歴を追跡できる
# MAGIC - 大量のデータを高速に処理できる
# MAGIC - データの整合性が保証される
# MAGIC
# MAGIC ### 処理内容
# MAGIC
# MAGIC 1. PandasのDataFrameをSparkのDataFrameに変換
# MAGIC 2. Deltaテーブルとして保存
# MAGIC 3. 変更データフィード（CDC）を有効化して、データの変更を追跡できるようにする

# COMMAND ----------

# CATALOGとSCHEMAが存在しない場合は作成
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# Pandas DataFrameをSpark DataFrameに変換
if not chunks_dataframe.empty:
    # データがある場合：Pandas DataFrameをそのまま変換
    spark_df = spark.createDataFrame(chunks_dataframe)
    print(f"Spark DataFrameを作成しました: {spark_df.count()} 行")
else:
    # データがない場合：空のDataFrameをスキーマ定義付きで作成
    schema = StructType([
        StructField("pdf_path", StringType(), True),      # PDFファイルのパス
        StructField("page_number", IntegerType(), True),  # ページ番号
        StructField("chunk_id", IntegerType(), True),     # チャンク番号
        StructField("chunk_text", StringType(), True),    # チャンクのテキスト
        StructField("id", IntegerType(), True)            # 一意のID
    ])
    spark_df = spark.createDataFrame([], schema)
    print("データが空のため、空のDataFrameを作成しました")

# Deltaテーブルに保存
table_name = f"{CATALOG}.{SCHEMA}.pdf_chunks"
print(f"Deltaテーブルに保存しています: {table_name}")

spark_df.write.format("delta").mode("overwrite").saveAsTable(table_name)

# 変更データフィード（CDC）を有効化
# これにより、テーブルの変更履歴を追跡できるようになります
spark.sql(f"ALTER TABLE {table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

print(f"✓ 保存完了: {table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ8: 保存したデータを確認する
# MAGIC
# MAGIC SQLを使って、Deltaテーブルに保存されたデータを確認します。
# MAGIC
# MAGIC ### 確認内容
# MAGIC
# MAGIC - 各チャンクのID
# MAGIC - PDFファイルのパス
# MAGIC - ページ番号
# MAGIC - チャンク番号
# MAGIC - テキストの最初の200文字（プレビュー）
# MAGIC
# MAGIC ### 注意
# MAGIC
# MAGIC 以下のSQLの `FROM` 句のテーブル名は、実際の環境に合わせて変更してください。

# COMMAND ----------

# 保存したチャンクデータを確認
sql = f"""SELECT 
  id,                                          -- チャンクID
  pdf_path,                                    -- PDFファイルのパス
  page_number,                                 -- ページ番号
  chunk_id,                                    -- チャンク番号
  substr(chunk_text, 1, 200) AS chunk_preview  -- テキストの最初の200文字
FROM {CATALOG}.{SCHEMA}.pdf_chunks
LIMIT 10;
"""
display(spark.sql(sql))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ9: ベクトル検索インデックスを作成する
# MAGIC
# MAGIC ### ベクトル検索とは
# MAGIC
# MAGIC ベクトル検索は、テキストの意味を理解して類似した内容を検索する技術です。
# MAGIC 従来のキーワード検索と違い、同じ言葉が含まれていなくても意味が似ていれば検索できます。
# MAGIC
# MAGIC ### 処理の流れ
# MAGIC
# MAGIC 1. **ベクトル検索エンドポイントの作成**  
# MAGIC    ベクトル検索を実行するためのサービスを起動します
# MAGIC
# MAGIC 2. **Delta Sync Indexの作成**  
# MAGIC    Deltaテーブルと連携して、自動的にベクトルインデックスを更新する仕組みを作ります
# MAGIC
# MAGIC ### パラメータの説明
# MAGIC
# MAGIC - **endpoint_name**: ベクトル検索サービスの名前
# MAGIC - **index_name**: 作成するインデックスの名前
# MAGIC - **primary_key**: テーブルの主キー（各行を一意に識別する列）
# MAGIC - **source_table_name**: データソースとなるDeltaテーブル
# MAGIC - **pipeline_type**: 更新方法（TRIGGERED = 手動更新、CONTINUOUS = 自動更新）
# MAGIC - **embedding_source_column**: ベクトル化する対象の列
# MAGIC - **embedding_model_endpoint_name**: 使用する埋め込みモデル

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
import time

# ベクトル検索クライアントを初期化
vsc = VectorSearchClient()

print("ステップ1: ベクトル検索エンドポイントの確認・作成")

# 既存のエンドポイント一覧を取得
existing_endpoints = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]

# エンドポイントが存在しない場合は作成
if VECTOR_SEARCH_ENDPOINT not in existing_endpoints:
    print(f"エンドポイント '{VECTOR_SEARCH_ENDPOINT}' を作成しています...")
    vsc.create_endpoint_and_wait(
        name=VECTOR_SEARCH_ENDPOINT, 
        endpoint_type="STANDARD"
    )
    print(f"✓ エンドポイント '{VECTOR_SEARCH_ENDPOINT}' を作成しました")
else:
    print(f"✓ エンドポイント '{VECTOR_SEARCH_ENDPOINT}' は既に存在します")

print("\nステップ2: ベクトル検索インデックスの作成")
print("この処理には数分かかる場合があります...")

# Delta Sync Indexを作成
try:
    response = vsc.create_delta_sync_index_and_wait(
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        index_name=VECTOR_INDEX_FULLNAME,
        primary_key="id",
        source_table_name=f"{CATALOG}.{SCHEMA}.pdf_chunks",
        pipeline_type="TRIGGERED",
        embedding_source_column="chunk_text",
        embedding_model_endpoint_name=EMBEDDING_MODEL_ENDPOINT
    )
    print(f"✓ インデックス '{VECTOR_INDEX_FULLNAME}' を作成しました")
    print(f"インデックス作成の詳細: {response}")
except Exception as e:
    print(f"エラーが発生しました: {e}")
    print("既にインデックスが存在する場合は、このエラーは無視できます")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ10: ベクトル検索を試してみる（オプション）
# MAGIC
# MAGIC 作成したインデックスを使って、実際に検索を試してみましょう。
# MAGIC
# MAGIC ### 検索の仕組み
# MAGIC
# MAGIC 1. 検索クエリ（質問文）をベクトルに変換
# MAGIC 2. インデックス内のチャンクベクトルと比較
# MAGIC 3. 最も類似度が高いチャンクを返す
# MAGIC
# MAGIC ### 使い方
# MAGIC
# MAGIC `query_text` の部分を自分の質問に変更して実行してください。

# COMMAND ----------

# 検索クエリ（質問文）
query_text = "休暇の申請方法について教えてください"

print(f"検索クエリ: {query_text}\n")

try:
    # ベクトル検索を実行
    results = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        index_name=VECTOR_INDEX_FULLNAME
    ).similarity_search(
        query_text=query_text,
        columns=["pdf_path", "page_number", "chunk_text"],
        num_results=3  # 上位3件を取得
    )
    
    print("検索結果:\n")
    
    # 結果を表示
    for i, result in enumerate(results.get("result", {}).get("data_array", []), 1):
        print(f"--- 結果 {i} ---")
        print(f"PDFファイル: {result[0]}")
        print(f"ページ番号: {result[1]}")
        print(f"テキスト: {result[2][:200]}...")  # 最初の200文字
        print()
        
except Exception as e:
    print(f"検索エラー: {e}")
    print("インデックスの作成が完了していない可能性があります。")
    print("数分待ってから再度実行してください。")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 補足: さらなる改善のアイデア
# MAGIC
# MAGIC このノートブックは基本的な実装ですが、以下のような改善が可能です：
# MAGIC
# MAGIC ### 1. OCR（光学文字認識）の追加
# MAGIC
# MAGIC **課題**: スキャンされたPDFは文字情報を持たないため、テキスト抽出できません
# MAGIC
# MAGIC **解決策**: 
# MAGIC - `pytesseract` や `easyocr` を使って画像から文字を認識
# MAGIC - Databricks上で動作するOCRサービスを利用
# MAGIC
# MAGIC ```python
# MAGIC # OCRの例（pytesseractを使用）
# MAGIC import pytesseract
# MAGIC from pdf2image import convert_from_path
# MAGIC
# MAGIC def extract_text_with_ocr(pdf_path):
# MAGIC     images = convert_from_path(pdf_path)
# MAGIC     text = ""
# MAGIC     for image in images:
# MAGIC         text += pytesseract.image_to_string(image, lang='jpn')
# MAGIC     return text
# MAGIC ```
# MAGIC
# MAGIC ### 2. 日本語PDFへの対応
# MAGIC
# MAGIC **課題**: 現在の設定は英語向けです
# MAGIC
# MAGIC **解決策**:
# MAGIC - 文分割器を日本語対応に変更: `pysbd.Segmenter(language="ja")`
# MAGIC - 日本語対応の埋め込みモデルを使用
# MAGIC - トークナイザーを日本語モデルに変更
# MAGIC
# MAGIC ### 3. チャンク分割の最適化
# MAGIC
# MAGIC **改善ポイント**:
# MAGIC - 段落単位での分割を優先
# MAGIC - 見出しを考慮した分割
# MAGIC - 表やリストの構造を保持
# MAGIC
# MAGIC ```python
# MAGIC # 段落を考慮した分割の例
# MAGIC def split_by_paragraphs(text):
# MAGIC     paragraphs = text.split('\n\n')
# MAGIC     return [p.strip() for p in paragraphs if p.strip()]
# MAGIC ```
# MAGIC
# MAGIC ### 4. メタデータの拡張
# MAGIC
# MAGIC **追加すると便利な情報**:
# MAGIC - ファイル名（パスから抽出）
# MAGIC - 処理日時
# MAGIC - PDFのバージョン情報
# MAGIC - セクション名や見出し
# MAGIC
# MAGIC ```python
# MAGIC # メタデータ追加の例
# MAGIC import datetime
# MAGIC from pathlib import Path
# MAGIC
# MAGIC metadata = {
# MAGIC     "filename": Path(pdf_path).name,
# MAGIC     "processed_at": datetime.datetime.now().isoformat(),
# MAGIC     "file_size": os.path.getsize(pdf_path)
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC ### 5. 大規模データの処理
# MAGIC
# MAGIC **課題**: 大量のPDFを処理するとメモリ不足になる可能性があります
# MAGIC
# MAGIC **解決策**:
# MAGIC - バッチ処理: PDFを小分けにして処理
# MAGIC - Spark UDFを使った並列処理
# MAGIC - ストリーミング処理の導入
# MAGIC
# MAGIC ```python
# MAGIC # バッチ処理の例
# MAGIC def process_pdfs_in_batches(pdf_list, batch_size=10):
# MAGIC     for i in range(0, len(pdf_list), batch_size):
# MAGIC         batch = pdf_list[i:i+batch_size]
# MAGIC         # バッチごとに処理
# MAGIC         process_batch(batch)
# MAGIC ```
# MAGIC
# MAGIC ### 6. エラーハンドリングの強化
# MAGIC
# MAGIC **改善ポイント**:
# MAGIC - 処理失敗したPDFのログ記録
# MAGIC - リトライ機能の追加
# MAGIC - エラー詳細の保存
# MAGIC
# MAGIC ```python
# MAGIC # エラーログの例
# MAGIC import logging
# MAGIC
# MAGIC logging.basicConfig(level=logging.INFO)
# MAGIC logger = logging.getLogger(__name__)
# MAGIC
# MAGIC try:
# MAGIC     process_pdf(pdf_path)
# MAGIC except Exception as e:
# MAGIC     logger.error(f"Failed to process {pdf_path}: {e}")
# MAGIC ```
# MAGIC
# MAGIC ### 7. インデックスの更新戦略
# MAGIC
# MAGIC **考慮事項**:
# MAGIC - 新しいPDFが追加されたときの差分更新
# MAGIC - 定期的な再インデックス
# MAGIC - バージョン管理
# MAGIC
# MAGIC ```python
# MAGIC # 差分更新の例
# MAGIC def update_index_incrementally():
# MAGIC     # 前回処理以降に追加されたファイルのみ処理
# MAGIC     new_files = get_new_files_since_last_run()
# MAGIC     process_new_files(new_files)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## まとめ
# MAGIC
# MAGIC このノートブックでは、以下のことを学びました：
# MAGIC
# MAGIC 1. ✓ PDFファイルからテキストを抽出する方法
# MAGIC 2. ✓ 長いテキストを適切な長さに分割（チャンク化）する方法
# MAGIC 3. ✓ データをDeltaテーブルに保存する方法
# MAGIC 4. ✓ ベクトル検索インデックスを作成する方法
# MAGIC 5. ✓ 実際に検索を実行する方法
# MAGIC
# MAGIC ### 次のステップ
# MAGIC
# MAGIC - 自分のPDFファイルで試してみる
# MAGIC - 検索クエリを変えて結果を確認する
# MAGIC - 補足セクションの改善アイデアを実装してみる
# MAGIC
# MAGIC ### トラブルシューティング
# MAGIC
# MAGIC **エラーが出た場合**:
# MAGIC 1. エラーメッセージをよく読む
# MAGIC 2. どのセルでエラーが出たか確認する
# MAGIC 3. 設定値（CATALOG、SCHEMA等）が正しいか確認する
# MAGIC 4. PDFファイルが正しい場所にあるか確認する
# MAGIC
# MAGIC **質問がある場合**:
# MAGIC - エラーメッセージ全文
# MAGIC - エラーが出たセルの番号
# MAGIC - 実行環境の情報
# MAGIC
# MAGIC を共有してください。

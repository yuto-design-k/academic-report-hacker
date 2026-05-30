import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
import os
import shutil

# アプリのタイトルと説明
st.set_page_config(page_title="大学レポート必勝構成案ジェネレーター", layout="wide")
st.title("🎓 大学レポート必勝構成案ジェネレーター (100%ローカル完結)")
st.write("「シラバス」「A判定レポート」「レジュメ」を元に、絶対に減点されない必勝の章立てを自動生成します。")

# 一時フォルダのクリーンアップ関数
def clear_old_data():
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
    if os.path.exists("./tmp"):
        shutil.rmtree("./tmp")
    os.makedirs("./tmp", exist_ok=True)

# 画面を2列に分割 (左: ファイルアップロード、右: 結果表示)
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📂 資料のアップロード")
    st.caption("※PDF形式のみ対応しています")
    
    syllabus_file = st.file_uploader("1. シラバス（評価基準）をアップロード", type=["pdf"])
    past_report_file = st.file_uploader("2. 過去のA判定レポート（成功例）をアップロード", type=["pdf"])
    resume_file = st.file_uploader("3. 今回の授業レジュメ・お題をアップロード", type=["pdf"])
    
    generate_btn = st.button("🚀 必勝構成案を生成する", type="primary")

with col2:
    st.header("📋 生成された必勝構成案")
    
    if generate_btn:
        if not (syllabus_file and past_report_file and resume_file):
            st.error("エラー: 3つのファイルをすべてアップロードしてください。")
        else:
            with st.spinner("ローカルAIが3つの資料をハック中...（数十秒かかります）"):
                try:
                    # 一時保存とデータクリア
                    clear_old_data()
                    
                    # 3つのファイルをローカルに保存して読み込む
                    all_docs = []
                    files = {
                        "Syllabus": syllabus_file,
                        "PastReport": past_report_file,
                        "Resume": resume_file
                    }
                    
                    for key, file_obj in files.items():
                        tmp_path = f"./tmp/{key}.pdf"
                        with open(tmp_path, "wb") as f:
                            f.write(file_obj.getbuffer())
                        
                        loader = PyPDFLoader(tmp_path)
                        loaded_docs = loader.load()
                        # どの資料のデータか判別できるようにメタデータを付与
                        for doc in loaded_docs:
                            doc.metadata["source_type"] = key
                        all_docs.extend(loaded_docs)
                    
                    # テキストを細かく分割
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
                    chunks = text_splitter.split_documents(all_docs)
                    
                    # ローカルDBと埋め込みモデルの初期化（Ollamaが起動している必要があります）
                    embeddings = OllamaEmbeddings(model="nomic-embed-text")
                    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
                    
                    # データベースからそれぞれの資料を検索して集約
                    # (本来は別々に検索しますが、プロトタイプとして全データをコンテキストに集約)
                    context_text = ""
                    for chunk in chunks:
                        context_text += f"\n[{chunk.metadata['source_type']}からの引用]:\n{chunk.page_content}\n"
                    
                    # AIへの厳格な命令（プロンプト）
                    prompt = f"""
                    あなたは大学のレポート評価をハックする天才教務アシスタントです。
                    提供されたデータ（シラバス、過去のA判定レポート、授業レジュメ）のみを根拠にして、今回の課題に対する「最高評価を狙える必勝構成案（章立てと執筆ガイド）」を作成してください。
                    
                    【提供された資料のデータ】
                    {context_text}
                    
                    【出力フォーマット】
                    以下の構成で、具体的に出力してください。
                    
                    1. 🚨 【絶対厳守】シラバスから抽出した減点トラップ（文字数、必須項目、提出ルールなど）
                    2. 💡 【加点ポイント】過去のA判定レポートから分析した、教授が好む文章のトーンや構成の癖
                    3. 📌 【必勝構成案】今回のレジュメの内容をベースに組み立てた、具体的な章立て（第1章〜第X章）と、各章に書くべきレジュメ上のキーワード・理論の指示
                    
                    一般論の知識は使わず、必ず資料内の言葉を使って具体的に執筆の道標を作ってください。
                    """
                    
                    # ローカルLLMで推論 (Llama3を使用)
                    llm = Ollama(model="llama3", temperature=0.2)
                    response = llm.invoke(prompt)
                    
                    # 結果を表示
                    st.success("構成案が完成しました！")
                    st.markdown(response)
                    
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")
                    st.info("※PCでOllamaアプリが起動しており、llama3 と nomic-embed-text のモデルがダウンロードされているか確認してください。")

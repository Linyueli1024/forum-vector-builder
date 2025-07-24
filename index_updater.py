import pandas as pd
import json
import numpy as np
import faiss
import pymysql
from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer
from apscheduler.schedulers.background import BackgroundScheduler
import time

# === 1. 初始化模型 ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === 2. MySQL连接信息 ===
DB_URI = "mysql+pymysql://root:Lyl%40123456%21@8.208.33.213:3306/sofu"
engine = create_engine(DB_URI)

def extract_question_text(content_str):
    try:
        obj = json.loads(content_str)
        blocks = obj.get("blocks", [])
        for block in blocks:
            if block.get("type") == "text":
                return block.get("data", "").strip()
    except:
        return ""
    return ""

def update_index():
    print("⏳ 正在拉取数据并更新索引...")

    # === 3. 读取数据库中的数据 ===
    questions = pd.read_sql("SELECT * FROM questions", engine)
    questions.columns = ["id", "content", "bonus", "user_id", "created_at", "updated_at", "same_ques", "title"]
    questions["question_text"] = questions["content"].apply(extract_question_text)

    answers = pd.read_sql(
    "SELECT * FROM answers WHERE target_type = 'question'", engine
)[["id", "target_id", "user_id", "text", "created_at", "updated_at", "likes"]]

    # 把 target_id 重命名为 question_id
    answers = answers.rename(columns={"target_id": "question_id"})

    # === 4. 每个问题拼接前2个回答 ===
    answer_group = answers.groupby("question_id")["text"].apply(
        lambda x: "\n".join([f"回答：{a.strip()}" for a in x.tolist()[:2]])
    ).reset_index()
    print(answer_group.columns.tolist())
    qa_df = pd.merge(questions[["id", "question_text"]], answer_group,
                     left_on="id", right_on="question_id", how="left")
    qa_df["qa_text"] = "问题：" + qa_df["question_text"] + "\n" + qa_df["text"].fillna("")
    qa_final = qa_df[["id", "qa_text"]].rename(columns={"id": "question_id"})

    # === 5. 生成向量并构建索引 ===
    embeddings = model.encode(qa_final["qa_text"].tolist(), normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings))

    # === 6. 保存索引和映射文件 ===
    faiss.write_index(index, "qa_index.faiss")
    records = qa_final.to_dict(orient="records")
    with open("qa_mapping.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print("✅ 索引和映射更新完成")

# === 先立即执行一次（首次启动时）===
update_index()
# === 7. 启动定时任务（每5分钟）===
scheduler = BackgroundScheduler()
scheduler.add_job(update_index, 'interval', minutes=5)
scheduler.start()

print("📡 索引更新服务启动中，按 Ctrl+C 停止")

# 保持进程运行
try:
    while True:
        time.sleep(60)
except (KeyboardInterrupt, SystemExit):
    scheduler.shutdown()

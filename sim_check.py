import os
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
import faiss
import sqlite3
from datetime import datetime

class WorkOrderDuplicationChecker:
    def __init__(self, model_name: str = "shibing624/text2vec-base-chinese", 
                 index_path: str = "work_order_index.faiss", 
                 db_path: str = "work_orders.db"):
        # 初始化语义模型
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # 初始化向量索引
        self.index_path = index_path
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # 使用内积索引，适合余弦相似度
        
        # 初始化数据库连接
        self.db_path = db_path
        self.init_db()
        
    def init_db(self) -> None:
        """初始化SQLite数据库"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS work_orders
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      order_id TEXT UNIQUE,
                      content TEXT,
                      embedding BLOB,
                      create_time TIMESTAMP)''')
        conn.commit()
        conn.close()
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本的向量表示"""
        return self.model.encode(text, convert_to_numpy=True)
    
    def _save_embedding_to_db(self, order_id: str, content: str, embedding: np.ndarray) -> None:
        """将工单信息和向量保存到数据库"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        embedding_bytes = embedding.tobytes()
        create_time = datetime.now()
        
        try:
            c.execute("INSERT INTO work_orders (order_id, content, embedding, create_time) VALUES (?, ?, ?, ?)",
                     (order_id, content, embedding_bytes, create_time))
            conn.commit()
        except sqlite3.IntegrityError:
            # 处理order_id重复的情况
            print(f"工单ID {order_id} 已存在，更新记录")
            c.execute("UPDATE work_orders SET content=?, embedding=?, create_time=? WHERE order_id=?",
                     (content, embedding_bytes, create_time, order_id))
            conn.commit()
        finally:
            conn.close()
    
    def _update_index(self, embedding: np.ndarray, order_id: str) -> None:
        """更新向量索引"""
        # 将向量添加到索引
        self.index.add(embedding.reshape(1, -1))
        # 保存索引到文件
        faiss.write_index(self.index, self.index_path)
    
    def add_work_order(self, order_id: str, content: str) -> None:
        """添加新工单到系统"""
        embedding = self._get_embedding(content)
        self._save_embedding_to_db(order_id, content, embedding)
        self._update_index(embedding, order_id)
    
    def check_duplication(self, content: str, top_k: int = 5, threshold: float = 0.8) -> List[Tuple[str, float]]:
        """检查新工单是否与历史工单重复
        
        Args:
            content: 新工单内容
            top_k: 返回最相似的k个历史工单
            threshold: 相似度阈值，超过此值被认为是重复工单
            
        Returns:
            包含(order_id, similarity)的列表，表示最相似的历史工单及其相似度
        """
        if self.index.ntotal == 0:
            return []
        
        # 获取新工单的向量表示
        query_embedding = self._get_embedding(content)
        
        # 在索引中查找最相似的历史工单
        distances, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        
        # 获取相似工单的order_id和相似度
        similar_orders = []
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # 索引中可能有空槽
                continue
                
            similarity = distances[0][i]
            # 余弦相似度 = 内积 / (模长乘积)，由于使用了归一化向量，内积即为余弦相似度
            if similarity >= threshold:
                c.execute("SELECT order_id FROM work_orders WHERE id=?", (idx + 1,))  # FAISS索引从0开始，数据库ID从1开始
                result = c.fetchone()
                if result:
                    similar_orders.append((result[0], similarity))
        
        conn.close()
        return sorted(similar_orders, key=lambda x: x[1], reverse=True)

# 使用示例
if __name__ == "__main__":
    # 初始化检测系统
    checker = WorkOrderDuplicationChecker()
    
    # 添加历史工单
    checker.add_work_order("WO2023001", "电站A变压器故障，无输出电压")
    checker.add_work_order("WO2023002", "电站B线路老化，存在安全隐患")
    checker.add_work_order("WO2023003", "电站C逆变器过热，频繁停机")
    
    # 检查新工单是否重复
    new_order_content = "电站A变压器无输出，怀疑是电压问题"
    duplicates = checker.check_duplication(new_order_content, threshold=0.7)
    
    if duplicates:
        print(f"发现相似工单，相似度: {duplicates[0][1]:.2f}")
        print(f"相似工单ID: {duplicates[0][0]}")
    else:
        print("未发现重复工单")    
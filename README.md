# rag-qa

基于 RAG（检索增强生成）的笔记问答系统，解决大模型幻觉问题。先检索相关笔记，再让大模型根据笔记回答，不靠模型记忆。

## 技术栈

Python / sentence-transformers / ChromaDB / 通义千问 API / OpenAI 兼容接口

## RAG 架构

**离线阶段（建库，只做一次）：**
```
文档 → 分块 → Embedding（文字变向量）→ 存入 ChromaDB
```

**在线阶段（每次提问时执行）：**
```
用户问题 → Embedding → 向量检索 Top-K → 组装 Prompt → 大模型回答
```

## 核心优势

- 解决幻觉问题：模型只根据检索到的资料回答，不靠训练记忆
- 支持私有数据：你的笔记、文档、公司内部资料，模型训练时没见过也能回答
- 防御性解析：`safe_json_parse()` 处理模型偶发的格式异常

## 问答示例
```
问题：json怎么解析？
回答：json.loads()把字符串转成字典，json.dumps()把字典转成字符串

问题：怎么处理大模型输出不稳定的问题？
回答：设置 temperature=0，保证大模型每次输出格式一致

问题：数据清洗怎么处理空值？
回答：先用 strip() 去除空格伪装的空值，再用中位数填充
```

## 快速开始
```bash
pip install sentence-transformers chromadb openai
python rag_demo.py        # 只检索，不调用大模型
python rag_with_ai.py     # 完整 RAG 链路，检索 + 大模型回答
```

## 项目结构
```
rag-qa/
├── rag_demo.py        # 离线建库 + 在线检索
├── rag_with_ai.py     # 完整 RAG 链路，接入大模型
└── safe_json_parse.py # 防御性 JSON 解析函数
```# rag-qa
基于RAG的笔记问答系统 | ChromaDB + Sentence-Transformers + 通义千问 | 解决大模型幻觉问题

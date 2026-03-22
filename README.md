# rag-qa

基于 RAG（检索增强生成）的笔记问答系统，解决大模型幻觉问题。提供两个版本：手写版和 LangChain 版。

## 技术栈

Python / sentence-transformers / ChromaDB / LangChain / 通义千问 API / OpenAI 兼容接口

## 两个版本对比

| | 手写版 | LangChain 版 |
|---|---|---|
| 文件 | rag_with_ai.py | langchain_rag.py |
| 检索 | 手动调用 ChromaDB | `vectorstore.as_retriever()` |
| 组装 Prompt | 手动拼字符串 | `ChatPromptTemplate` |
| 调用模型 | `client.chat.completions.create()` | `ChatOpenAI` + `|` 管道符 |
| 代码风格 | 逐步手写，易理解 | 链式调用，易扩展 |

## RAG 架构

**离线阶段（建库，只做一次）：**
```
文档 → Embedding（文字变向量）→ 存入 ChromaDB
```

**在线阶段（每次提问时执行）：**
```
用户问题 → 向量检索 Top-K → 组装 Prompt → 大模型回答
```

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
pip install sentence-transformers chromadb openai langchain langchain-community langchain-openai
python rag_with_ai.py     # 手写版：完整 RAG 链路
python langchain_rag.py   # LangChain 版：链式调用
```

## LangChain 核心概念

- `|` 管道符：把上一步结果传给下一步，像流水线
- `ChatPromptTemplate`：标准化 Prompt 模板管理
- `StrOutputParser()`：从模型返回对象里只取纯文字
- `as_retriever()`：把向量库变成标准检索器组件

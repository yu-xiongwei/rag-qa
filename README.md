# RAG 笔记问答系统

> 用你自己的笔记回答问题，模型只说笔记里有的内容，杜绝幻觉。

## 解决什么问题

直接问大模型"我笔记里怎么写的"，它会编造内容。
本项目基于 RAG 架构，让模型只根据你的真实笔记回答，检索不到就明确说不知道。

## 系统架构

离线建库（只做一次）：
笔记文本 → sentence-transformers 向量化 → 存入 ChromaDB

在线问答（每次提问）：
用户问题 → 向量化 → ChromaDB Top-K 检索 → 组装 Prompt → 通义千问 → 回答

## 效果展示

问：json 怎么解析？
答：json.loads() 把字符串转成字典，json.dumps() 把字典转成字符串

问：怎么处理大模型输出不稳定？
答：设置 temperature=0，保证每次输出格式一致

问：什么是 RAG？（笔记里没有）
答：根据提供的资料，我无法回答这个问题。

## 两个版本对比

|  | 手写版 rag_with_ai.py | LangChain 版 langchain_rag.py |
|--|--|--|
| 适合场景 | 理解原理，逐步调试 | 生产扩展，换模型方便 |
| 检索方式 | 手动调用 ChromaDB | vectorstore.as_retriever() |
| Prompt 组装 | 手动拼字符串 | ChatPromptTemplate |
| 输出解析 | 手动取 .content | StrOutputParser() |
| 换模型成本 | 改 base_url | 改一行配置 |

## 快速开始

pip install sentence-transformers chromadb openai langchain langchain-community langchain-openai

export DASHSCOPE_API_KEY=your_key_here   # 不要硬编码 key

python rag_with_ai.py      # 手写版：完整 RAG 链路，看清楚每一步
python langchain_rag.py    # LangChain 版：链式调用，生产级写法

## 工程细节

- temperature=0：保证每次检索 + 回答结果可复现
- Top-K 设为 3：避免无关内容干扰模型判断
- 检索不到相关内容时，Prompt 明确指示模型拒绝回答，而不是编造
- LangChain 管道符 |：retriever → prompt → model → output_parser，换模型只改一行

## 技术栈

Python / sentence-transformers / ChromaDB / LangChain / 通义千问 API

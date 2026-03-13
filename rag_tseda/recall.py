"""T-SEDA RAG 召回与标注（从 text_annotation 拷贝并改为使用项目内路径与 LLM）"""
import os
from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.llm.llm_client import get_llm

# 本模块所在目录，用于解析 CSV 与向量库路径
_RAG_TSEDA_DIR = Path(__file__).resolve().parent
_CSV_PATH = _RAG_TSEDA_DIR / "data" / "T_SEDA_label_methods.csv"
_VECTOR_STORE_DIR = _RAG_TSEDA_DIR / "vector_store_tseda"


def _get_embeddings():
    """延迟创建 DashScope embedding，与 get_llm 共用同一 env。"""
    from langchain_community.embeddings import DashScopeEmbeddings

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("未设置 DASHSCOPE_API_KEY，请在 config/llm_model.env 中配置")
    return DashScopeEmbeddings(
        model="text-embedding-v4",
        dashscope_api_key=api_key,
    )


def tseda_csv_to_documents(csv_path: Path | None = None):
    """将 T-SEDA 标签 CSV 转为 LangChain Document 列表。"""
    path = csv_path or _CSV_PATH
    df = pd.read_csv(path, encoding="utf-8-sig")
    documents = []
    for idx, row in df.iterrows():
        category = str(row.get("对话代码类别", "")).strip()
        code = str(row.get("唯一代码", "")).strip()
        strategy = str(row.get("作用和策略", "")).strip()
        keywords = str(row.get("举例子", "")).strip()
        content_parts = [
            f"标签类别: {category}",
            f"唯一代码: {code}",
            f"作用和策略: {strategy}",
            f"关键词示例: {keywords}",
        ]
        text = "\n".join(content_parts)
        metadata = {
            "row_index": idx,
            "source": str(path),
            "标签类别": category,
            "唯一代码": code,
            "作用和策略": strategy,
            "关键词": keywords,
        }
        documents.append(Document(page_content=text, metadata=metadata))
    return documents


def create_or_load_vector_store(
    documents=None,
    persist_dir: Path | None = None,
    force_recreate: bool = False,
):
    """创建或加载 FAISS 向量存储。"""
    persist_dir = persist_dir or _VECTOR_STORE_DIR
    persist_dir = Path(persist_dir)
    if documents is None:
        documents = tseda_csv_to_documents()
    embeddings = _get_embeddings()
    if persist_dir.exists() and not force_recreate:
        try:
            return FAISS.load_local(
                str(persist_dir),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            pass
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
        length_function=len,
    )
    splits = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(splits, embeddings)
    persist_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(persist_dir))
    return vectorstore


async def create_tseda_rag_chain(vectorstore=None):
    """创建 T-SEDA RAG 链。"""
    if vectorstore is None:
        vectorstore = create_or_load_vector_store()
    from rag_tseda.prompt import TSEDA_PROMPT

    llm = get_llm(model="qwen-plus", streaming=False)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["spoken_content"]))
        )
        | TSEDA_PROMPT
        | llm
        | StrOutputParser()
    )

    async def invoke(args):
        spoken_content = args.get("spoken_content")
        context_info = args.get("context_info", "")
        answer = await chain.ainvoke({
            "spoken_content": spoken_content,
            "context_info": context_info,
        })
        return {"result": answer}

    return invoke


async def label_by_rag(spoken_content: str, context_info: str = ""):
    """使用 RAG 进行单条标注。返回 dict 含 result。"""
    vectorstore = create_or_load_vector_store()
    invoke = await create_tseda_rag_chain(vectorstore)
    return await invoke({
        "spoken_content": spoken_content,
        "context_info": context_info,
    })

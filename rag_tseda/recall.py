"""T-SEDA RAG 召回与标注（从 text_annotation 拷贝并改为使用项目内路径与 LLM）"""
import json
import os
from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.llm.llm_client import get_llm

# 本模块所在目录，用于解析 CSV/JSON 与向量库路径
_RAG_TSEDA_DIR = Path(__file__).resolve().parent
_CSV_PATH = _RAG_TSEDA_DIR / "data" / "T_SEDA_label_methods.csv"
_SCHEMA_PATH = _RAG_TSEDA_DIR / "data" / "metrics_output_schema.json"
_EVAL_STEPS_PATH = _RAG_TSEDA_DIR / "data" / "eval_steps.md"
_VECTOR_STORE_DIR = _RAG_TSEDA_DIR / "vector_store_tseda"


def _get_embeddings():
    """延迟创建 DashScope embedding，与 get_llm 共用同一 env。"""
    # 配置embedding模型
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


def _flatten_schema_node(obj, prefix: str = "") -> list[tuple[str, str]]:
    """将 schema 中某节点展平为 (字段路径, 描述) 列表，跳过 _ 开头的键。"""
    out = []
    if not isinstance(obj, dict):
        return out
    for k, v in obj.items():
        if k.startswith("_"):
            if k == "_description" and isinstance(v, str):
                out.append((prefix.rstrip("."), v))
            elif k == "_item" and isinstance(v, dict):
                out.extend(_flatten_schema_node(v, prefix))
            continue
        path = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            out.extend(_flatten_schema_node(v, path + "."))
        elif isinstance(v, str):
            out.append((path, v))
    return out


def metrics_schema_to_documents(schema_path: Path | None = None) -> list[Document]:
    """将 metrics_output_schema.json 转为 LangChain Document 列表，供评价 RAG 召回。"""
    path = schema_path or _SCHEMA_PATH
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lines = ["指标输出结构说明（POST /metrics 返回）：字段含义与评价可参考以下定义。"]
    for path_str, desc in _flatten_schema_node(data):
        lines.append(f"{path_str}: {desc}")
    text = "\n".join(lines)
    return [
        Document(
            page_content=text,
            metadata={"source": str(path), "type": "metrics_schema"},
        )
    ]


def eval_steps_to_documents(md_path: Path | None = None) -> list[Document]:
    """将 eval_steps.md（评价文案生成规范）转为 LangChain Document，供评价 RAG 召回。"""
    path = md_path or _EVAL_STEPS_PATH
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return [
        Document(
            page_content=text,
            metadata={"source": str(path), "type": "eval_steps"},
        )
    ]


def create_or_load_vector_store(
    documents=None,
    persist_dir: Path | None = None,
    force_recreate: bool = False,
):
    """创建或加载 FAISS 向量存储。默认包含 T-SEDA 标签 CSV、metrics_output_schema.json、eval_steps.md。若此前已持久化过向量库，需传 force_recreate=True 或删除 vector_store_tseda 目录后重建，才会包含新增文档。"""
    persist_dir = persist_dir or _VECTOR_STORE_DIR
    persist_dir = Path(persist_dir)
    if documents is None:
        documents = (
            tseda_csv_to_documents()
            + metrics_schema_to_documents()
            + eval_steps_to_documents()
        )
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


def _build_eval_query(metrics_dict: dict) -> str:
    """从指标 JSON 构建用于 FAISS 检索的查询文本，便于召回 T-SEDA 标签与策略相关文档。"""
    parts = [
        "T-SEDA 课堂讨论 评价 认知深度 探究 协作 表达 推理论证 质疑 补充 协调整合 知识联系 反思 引导",
    ]
    summary = metrics_dict.get("summary") or {}
    group = metrics_dict.get("group") or {}
    participants = metrics_dict.get("participants") or []
    if summary:
        parts.append(f"发言总数 {summary.get('total_utterances')} 参与人数 {summary.get('total_participants')}")
    if group:
        counts = group.get("label_counts") or {}
        if counts:
            parts.append("编码分布 " + " ".join(f"{k}{v}" for k, v in sorted(counts.items())))
    for p in participants[:5]:
        name = p.get("speaker") or p.get("student_id") or ""
        if name:
            parts.append(name)
        cog = p.get("cognitive") or {}
        if cog:
            parts.append("CDI IDI BE KCI CCI MDI")
    return " ".join(parts)


def get_eval_rag_context(metrics_dict: dict, k: int = 5) -> str:
    """
    使用 FAISS 向量库（与 T-SEDA 标注共用）检索与当前指标相关的文档，作为评价时的参考 context。
    调用方若在异步上下文中，建议用 asyncio.to_thread(get_eval_rag_context, metrics_dict, k) 避免阻塞。
    """
    vectorstore = create_or_load_vector_store()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    query = _build_eval_query(metrics_dict)
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs) if docs else ""

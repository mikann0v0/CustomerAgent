import ast
import base64
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI


CHAT_MODEL = "stepfun/step-3.5-flash:free"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
VISION_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

OPENROUTER_API_KEY = ""
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SILICON_API_KEY = ""
SILICON_BASE_URL = "https://api.siliconflow.com/v1"

class MultimodalCustomerAgent:
    def __init__(
        self,
        docs_dir: Optional[str] = None,
        pics_dir: Optional[str] = None,
        persist_dir: Optional[str] = None,
    ):
        base_dir = Path(__file__).resolve().parents[1]
        self.docs_dir = Path(docs_dir) if docs_dir else base_dir / "docs"
        self.pics_dir = Path(pics_dir) if pics_dir else self.docs_dir / "pics"
        self.persist_dir = Path(persist_dir) if persist_dir else Path(__file__).resolve().parent / "chroma_db"

        self.llm_model = CHAT_MODEL
        self.llm_temperature = 0.1
        self.llm_client = None
        # Use OpenRouter for Chat if Key is provided, else fallback to SiliconFlow chat models
        if OPENROUTER_API_KEY and OPENROUTER_API_KEY != "TO_BE_FILLED_LATER":
            self.llm_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
        else:
            self.llm_client = OpenAI(api_key=SILICON_API_KEY, base_url=SILICON_BASE_URL)
            self.llm_model = "Qwen/Qwen2.5-7B-Instruct"

        self.vision_model = VISION_MODEL
        self.vision_client = OpenAI(api_key=SILICON_API_KEY, base_url=SILICON_BASE_URL)

        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=SILICON_API_KEY,
            base_url=SILICON_BASE_URL,
        )

        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.qdrant_client = QdrantClient(path=str(self.persist_dir))

        if not self.qdrant_client.collection_exists("manual_multimodal_kb"):
            self.qdrant_client.create_collection(
                collection_name="manual_multimodal_kb",
                vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
            )
        # Note: Do not delete existing collection to preserve local cache
        # else:
        #     try:
        #         self.qdrant_client.delete_collection("manual_multimodal_kb")
        #         self.qdrant_client.create_collection(
        #             collection_name="manual_multimodal_kb",
        #             vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
        #         )
        #     except Exception:
        #         pass

        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name="manual_multimodal_kb",
            embedding=self.embeddings,
        )

        self.local_docs: List[Document] = []
        self.image_name_to_path = self._build_image_lookup()

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _summarize_image_for_match(
        self,
        image_path: str,
        image_id: str,
        manual_name: str,
        manual_text: str = "",
    ) -> str:
        if not image_path or not Path(image_path).exists():
            return f"{manual_name} 图片 {image_id}"

        if self.vision_client is None:
            return f"{manual_name} 图片 {image_id}"

        try:
            suffix = Path(image_path).suffix.lower().replace(".", "") or "jpeg"
            if suffix == "jpg":
                suffix = "jpeg"
            image_bytes = Path(image_path).read_bytes()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            data_url = f"data:image/{suffix};base64,{image_base64}"

            # 截取说明书前一部分做上下文
            context = self._normalize_text(manual_text)[:2000]

            response = self.vision_client.chat.completions.create(
                model=self.vision_model,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": "你是产品手册的视觉助手。请结合给定的产品说明书上下文，简要总结这张图片的主体对象以及对象正在做什么（或者处于什么状态）。不需要详细描述所有部件，只需抓住核心动作或状态即可，不超过50个字。",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"产品手册参考上下文：\n{context}\n\n请结合上下文，详细描述这张图片："},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
            )
            summary = self._normalize_text(response.choices[0].message.content or "")
            print(f"Generated summary for {image_id}: {summary}")
            return summary if summary else f"{manual_name} 图片 {image_id}"
        except Exception as e:
            print(f"Vision API error for {image_id}: {e}")
            return f"{manual_name} 图片 {image_id}"

    def _build_image_lookup(self) -> Dict[str, str]:
        image_lookup: Dict[str, str] = {}
        if not self.pics_dir.exists():
            return image_lookup

        for image_file in self.pics_dir.iterdir():
            if image_file.is_file():
                image_lookup[image_file.stem.lower()] = str(image_file)
        return image_lookup

    def _build_doc_id(self, item: Dict[str, Any], source_type: str) -> str:
        import uuid
        raw = json.dumps({"source_type": source_type, "item": item}, ensure_ascii=False, sort_keys=True)
        # Use UUID5 based on hash so it's deterministic but conforms to UUID standard required by Qdrant
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, raw))

    def _read_manual_file(self, file_path: Path) -> Tuple[str, List[str]]:
        content = file_path.read_text(encoding="utf-8")

        # 手册文件通常是 [正文, 图片ID列表] 的 Python 字面量结构。
        try:
            parsed = ast.literal_eval(content)
            if (
                isinstance(parsed, list)
                and len(parsed) == 2
                and isinstance(parsed[0], str)
                and isinstance(parsed[1], list)
            ):
                image_ids = [str(x) for x in parsed[1]]
                return parsed[0], image_ids
        except Exception:
            pass

        return content, []

    def _chunk_text(self, text: str, max_chars: int = 700, overlap: int = 120) -> List[str]:
        normalized = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not normalized:
            return []

        paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
        chunks: List[str] = []
        current = ""

        for para in paragraphs:
            candidate = f"{current}\n\n{para}" if current else para
            if len(candidate) <= max_chars:
                current = candidate
                continue

            if current:
                chunks.append(current)
            if len(para) <= max_chars:
                current = para
            else:
                start = 0
                while start < len(para):
                    end = start + max_chars
                    piece = para[start:end]
                    if piece.strip():
                        chunks.append(piece.strip())
                    if end >= len(para):
                        break
                    start = max(0, end - overlap)
                current = ""

        if current:
            chunks.append(current)

        return chunks

    def _extract_image_anchors(self, text: str, image_ids: List[str]) -> List[Dict[str, Any]]:
        anchors: List[Dict[str, Any]] = []
        positions = [m.start() for m in re.finditer(r"<PIC>", text)]
        for i, pos in enumerate(positions):
            image_id = image_ids[i] if i < len(image_ids) else None
            if not image_id:
                continue

            left = max(0, pos - 180)
            right = min(len(text), pos + 180)
            context = text[left:right].replace("\n", " ")
            context = re.sub(r"\s+", " ", context).strip()

            anchors.append({"image_id": image_id, "context": context})

        return anchors

    def _to_docs(self) -> Tuple[List[Document], List[str]]:
        docs: List[Document] = []
        ids: List[str] = []

        manual_files = sorted([p for p in self.docs_dir.glob("*.txt") if p.is_file()])

        # 为了测试，我们只过滤出包含图片最少的手册（VR头显手册.txt）
        # manual_files = [p for p in manual_files if p.name == "VR头显手册.txt"]

        for manual_file in manual_files:
            text, image_ids = self._read_manual_file(manual_file)
            manual_name = manual_file.stem

            clean_text = text.replace("<PIC>", " ")
            for chunk_idx, chunk in enumerate(self._chunk_text(clean_text)):
                item = {
                    "manual": manual_name,
                    "chunk_idx": chunk_idx,
                    "content": chunk,
                }
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "type": "text",
                            "manual": manual_name,
                            "chunk_idx": chunk_idx,
                        },
                    )
                )
                ids.append(self._build_doc_id(item, "text"))

            # 处理该手册下的所有图片，生成总结并入库
            for img_idx, image_id in enumerate(image_ids):
                image_path = self.image_name_to_path.get(image_id.lower(), "")

                # 调用 Vision 模型，结合文档全文生成图片描述
                image_summary = self._summarize_image_for_match(
                    image_path=image_path,
                    image_id=image_id,
                    manual_name=manual_name,
                    manual_text=text,
                )

                item = {
                    "manual": manual_name,
                    "image_idx": img_idx,
                    "image_id": image_id,
                    "content": image_summary,
                }

                # 将图片描述作为文档存入向量库
                docs.append(
                    Document(
                        page_content=image_summary,
                        metadata={
                            "type": "image_summary",
                            "manual": manual_name,
                            "image_id": image_id,
                            "image_path": image_path,
                        },
                    )
                )
                ids.append(self._build_doc_id(item, "image_summary"))

        return docs, ids

    def build_knowledge_base(self) -> Dict[str, int]:
        docs, ids = self._to_docs()
        if not docs:
            return {"docs": 0, "added": 0}

        self.local_docs = docs

        if self.vector_store is None:
            return {"docs": len(docs), "added": len(docs)}

        # Add all documents (Qdrant handles upserts based on IDs if provided)
        # Note: Langchain's QdrantVectorStore add_documents handles ID deduplication
        try:
            self.vector_store.add_documents(docs, ids=ids)
            added_count = len(docs) # Assuming all are upserted or added
        except Exception as e:
            print(f"Error adding documents to Qdrant: {e}")
            added_count = 0

        return {"docs": len(docs), "added": added_count}

    def _local_retrieve(self, query: str, top_k: int) -> List[Document]:
        if not self.local_docs:
            return []

        def build_terms(text: str) -> Tuple[set, set]:
            base_terms = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text.lower()))
            gram_terms = set()
            for term in list(base_terms):
                # 中文短语做双字切分，提升无空格中文检索命中率。
                if re.fullmatch(r"[\u4e00-\u9fff]+", term) and len(term) >= 2:
                    for i in range(len(term) - 1):
                        gram_terms.add(term[i : i + 2])
            return base_terms, gram_terms

        query_terms, query_grams = build_terms(query)
        if not query_terms and not query_grams:
            query_terms = {query.lower()}

        scored: List[Tuple[int, int, Document]] = []
        for idx, doc in enumerate(self.local_docs):
            text = doc.page_content.lower()
            score = 0
            score += 4 * sum(1 for t in query_terms if len(t) >= 2 and t in text)
            score += 1 * sum(1 for g in query_grams if g in text)

            manual_name = str(doc.metadata.get("manual", "")).lower()
            score += 2 * sum(1 for t in query_terms if len(t) >= 2 and t in manual_name)

            if score > 0:
                scored.append((score, idx, doc))

        if not scored:
            return self.local_docs[:top_k]

        scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        return [item[2] for item in scored[:top_k]]

    def _extractive_answer(self, question: str, context_lines: List[str], image_count: int) -> str:
        lines = [
            f"问题：{question}",
            "根据检索到的手册内容，相关信息如下：",
        ]
        for i, line in enumerate(context_lines[:4], start=1):
            safe_line = line.replace("<PIC>", "[图]")
            lines.append(f"{i}. {safe_line}")

        if image_count > 0:
            lines.append("可参考相关图片位置 <PIC>。")
        return "\n".join(lines)

    def analyze_intent(self, question: str) -> Dict[str, Any]:
        """使用 LLM 识别用户意图，分辨是否需要查阅手册。"""
        if not self.llm_client:
            return {"intent": "manual_qa", "reason": "No LLM available, default to manual QA"}

        system_prompt = """
        系统提示：
        你是智能客服意图识别专家。你需要分析用户的问题，判断其真实意图，并严格返回以下 JSON 格式。

        意图分类仅限以下四种：
        1. "manual_qa": 用户在询问某个电器的使用方法、部件说明、故障排查、操作步骤、维修建议、安全说明等需要查阅产品说明书才能回答的问题（例如：“如何启动发电机？”、“空调清洗滤网步骤”）。
        2. "after_sales": 用户在询问退换货政策、维修流程、商品损坏理赔、开发票、发票抬头修改、售后服务态度投诉等售后相关问题（例如：“支持7天无理由退换吗”、“发票类型是什么”）。
        3. "logistics_or_order": 用户在询问物流进度、少发漏发、快递员态度差、送错货等订单与物流相关问题（例如：“快递丢失了怎么办”、“少发了一件货”）。
        4. "complaint_or_other": 用户纯粹的谩骂、情绪宣泄，或者与产品、订单无关的闲聊。

        【注意】如果用户的问题虽然提到了家电，但是其核心诉求是退款、换货、包装破损理赔等，属于 "after_sales"。只有关于产品本身的技术细节或使用操作，才属于 "manual_qa"。

        返回格式必须是合法的 JSON，形如：
        {
            "intent": "manual_qa",
            "reason": "简短的一句话理由"
        }
        """
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                temperature=0.0,
                response_format={"type": "json_object"} if "qwen" not in self.llm_model.lower() else None,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
            )
            content = response.choices[0].message.content or "{}"

            # 手动提取 JSON，兼容不支持 JSON mode 的模型
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                content = match.group(0)
            return json.loads(content)
        except Exception as e:
            print(f"Intent Analysis Error: {e}")
            return {"intent": "manual_qa", "reason": "Error during analysis, default to manual QA"}

    def analyze_and_answer(self, question: str, top_k: int = 8) -> Dict[str, Any]:
        # 1. 意图识别
        intent_info = self.analyze_intent(question)
        intent = intent_info.get("intent", "manual_qa")
        print(f"Detected Intent: {intent} (Reason: {intent_info.get('reason', '')})")

        # 2. 针对非手册问答意图进行快速回复
        if intent == "logistics_or_order":
            return {
                "ret": "您好！关于您的订单物流或少发漏发问题，请提供您的订单号和快递单号，我将为您核实物流状态或安排补发/转交人工客服处理。",
                "image_list": [],
                "image_paths": [],
                "references": [],
            }
        elif intent == "after_sales":
            return {
                "ret": "您好！关于退换货、发票开具、售后维修或商品损坏等问题，为了更好地帮助您，请提供您的订单号以及相关照片凭证，我将为您转接售后专员处理。",
                "image_list": [],
                "image_paths": [],
                "references": [],
            }
        elif intent == "complaint_or_other":
            return {
                "ret": "非常抱歉给您带来不便。如果您对我们的服务或快递人员有任何不满，请留下您的联系方式和订单号，我将立刻为您记录并反馈给高级客服专员跟进处理。",
                "image_list": [],
                "image_paths": [],
                "references": [],
            }

        # 3. 如果是 manual_qa，走原有的检索和生成逻辑
        if self.vector_store is not None:
            from qdrant_client.http import models as rest
            
            # Detect keywords in the question to filter by specific manual
            # 常见的产品名称，映射到具体的手册名称
            keyword_to_manual = {
                "洗碗机": "洗碗机手册",
                "空调": "空调手册",
                "微波炉": "微波炉手册",
                "冰箱": "冰箱手册",
                "烤箱": "烤箱手册",
                "空气净化器": "空气净化器手册",
                "相机": "相机手册",
                "发电机": "发电机手册",
                "电钻": "电钻手册",
                "吹风机": "吹风机手册",
                "水泵": "水泵手册",
                "VR": "VR头显手册",
                "键盘": "功能键盘手册",
                "摩托艇": "摩托艇手册",
                "温控器": "可编程温控器手册",
                "人体工学椅": "人体工学椅手册",
                "健身单车": "健身单车手册",
                "追踪器": "健身追踪器手册",
                "摩托车": "儿童电动摩托车手册",
                "鼠标": "蓝牙激光鼠标手册",
                "蒸汽清洁机": "蒸汽清洁机手册"
            }
            
            target_manuals = []
            for keyword, manual_name in keyword_to_manual.items():
                if keyword in question:
                    target_manuals.append(manual_name)
                    
            if target_manuals:
                print(f"Detected keywords! Filtering retrieval to manuals: {target_manuals}")
                
            text_must_conditions = [
                rest.FieldCondition(
                    key="metadata.type",
                    match=rest.MatchValue(value="text"),
                )
            ]
            
            image_must_conditions = [
                rest.FieldCondition(
                    key="metadata.type",
                    match=rest.MatchValue(value="image_anchor"),
                )
            ]
            
            if target_manuals:
                manual_condition = rest.FieldCondition(
                    key="metadata.manual",
                    match=rest.MatchAny(any=target_manuals),
                )
                text_must_conditions.append(manual_condition)

            # 1. 一次性检索所有的 chunk 和 image_summary
            try:
                filter_obj = rest.Filter(must=text_must_conditions)
                # 不再强制过滤 "text" type，而是把 text_must_conditions 里的 type 过滤去掉，让它同时搜 text 和 image_summary
                any_type_conditions = [c for c in text_must_conditions if getattr(c, "key", "") != "metadata.type"]

                # We want to retrieve both type="text" and type="image_summary"
                type_condition = rest.FieldCondition(
                    key="metadata.type",
                    match=rest.MatchAny(any=["text", "image_summary"]),
                )
                any_type_conditions.append(type_condition)

                final_filter = rest.Filter(must=any_type_conditions)

                retrieved_docs = self.vector_store.similarity_search(
                    query=question,
                    k=top_k * 2, # 扩大检索量，确保图文都能被召回
                    filter=final_filter,
                )
            except Exception as e:
                print(f"Error filtering: {e}")
                retrieved_docs = self.vector_store.similarity_search(query=question, k=top_k * 2)

        else:
            retrieved_docs = self._local_retrieve(question, top_k * 2)

        context_lines: List[str] = []
        retrieved_images: List[Dict[str, str]] = []
        seen_image_ids = set()

        for i, doc in enumerate(retrieved_docs, start=1):
            m = doc.metadata
            doc_type = m.get("type", "text")
            manual = m.get("manual", "未知手册")

            if doc_type == "text":
                snippet = doc.page_content.strip()
                context_lines.append(f"[参考{i}][文本][{manual}] {snippet}")
            elif doc_type == "image_summary":
                # 如果召回了图片总结，说明该图片与问题高度相关
                snippet = doc.page_content.strip()
                image_id = m.get("image_id")
                image_path = m.get("image_path", "")

                context_lines.append(f"[参考{i}][图片总结][{manual}] 图片ID: {image_id}。内容描述: {snippet}")

                if image_id and image_id not in seen_image_ids:
                    seen_image_ids.add(image_id)
                    retrieved_images.append({
                        "image_id": image_id,
                        "image_path": image_path,
                    })

        context_text = "\n".join(context_lines)

        system_prompt = """
        你是客服知识库问答助手。
        请遵循以下规则：
        1. 仅依据提供的参考知识作答，不要编造。
        2. 回答简洁清晰，优先给步骤化结论。
        3. 如果参考知识不足，明确说明“资料中未明确提及”。
        4. 参考知识中包含“图片总结”，如果你的回答涉及到了这些图片的内容，请在回答的对应位置标注（如图：图片ID）。

        [参考知识]
        {context}
        """

        llm_client = self.llm_client
        if llm_client is None and SILICON_API_KEY:
            # Fallback to SILICON for chat if OPENROUTER is missing
            try:
                llm_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
                self.llm_model = CHAT_MODEL
            except Exception:
                pass

        used_llm = llm_client is not None

        if used_llm:
            try:
                response = llm_client.chat.completions.create(
                    model=self.llm_model,
                    temperature=self.llm_temperature,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt.format(context=context_text),
                        },
                        {"role": "user", "content": question},
                    ],
                )
                answer = response.choices[0].message.content or ""
                # Print debug statement to verify it's working
                print(f"DEBUG: Successfully invoked LLM {self.llm_model}")
            except Exception as e:
                print(f"LLM API Error: {e}")
                answer = self._extractive_answer(question, context_lines, 0)
        else:
            print("DEBUG: Using extractive answer fallback")
            answer = self._extractive_answer(question, context_lines, 0)

        return {
            "ret": answer,
            "image_list": [img["image_id"] for img in retrieved_images],
            "image_paths": [img["image_path"] for img in retrieved_images],
            "references": context_lines,
        }


if __name__ == "__main__":
    agent = MultimodalCustomerAgent()
    stats = agent.build_knowledge_base()
    print(f"知识库构建完成，总文档片段={stats['docs']}，新增={stats['added']}")

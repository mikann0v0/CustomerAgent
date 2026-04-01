import ast
import base64
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

class KnowledgeBaseManager:
    def __init__(
        self,
        docs_dir: Path,
        pics_dir: Path,
        persist_dir: Path,
        embeddings: Any,
        vision_client: Any,
        vision_model: str
    ):
        self.docs_dir = docs_dir
        self.pics_dir = pics_dir
        self.persist_dir = persist_dir
        self.embeddings = embeddings
        self.vision_client = vision_client
        self.vision_model = vision_model

        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.qdrant_client = QdrantClient(path=str(self.persist_dir))

        if not self.qdrant_client.collection_exists("manual_multimodal_kb"):
            self.qdrant_client.create_collection(
                collection_name="manual_multimodal_kb",
                vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
            )

        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name="manual_multimodal_kb",
            embedding=self.embeddings,
        )

        self.local_docs: List[Document] = []
        self.image_name_to_path = self._build_image_lookup()

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _build_image_lookup(self) -> Dict[str, str]:
        image_lookup: Dict[str, str] = {}
        if not self.pics_dir.exists():
            return image_lookup
        for image_file in self.pics_dir.iterdir():
            if image_file.is_file():
                image_lookup[image_file.stem.lower()] = str(image_file)
        return image_lookup

    def _summarize_image_for_match(
        self,
        image_path: str,
        image_id: str,
        manual_name: str,
        manual_text: str = "",
    ) -> str:
        if not image_path or not Path(image_path).exists():
            return f"{manual_name} 图片 {image_id}"

        try:
            suffix = Path(image_path).suffix.lower().replace(".", "") or "jpeg"
            if suffix == "jpg":
                suffix = "jpeg"
            image_bytes = Path(image_path).read_bytes()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            data_url = f"data:image/{suffix};base64,{image_base64}"
            context = self._normalize_text(manual_text)[:2000]

            response = self.vision_client.chat.completions.create(
                model=self.vision_model,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": "你是产品手册的视觉助手。请结合给定的产品说明书上下文，简要总结这张图片的主体对象以及对象正在做什么（或者处于什么状态）。不超过50个字。",
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
            return summary if summary else f"{manual_name} 图片 {image_id}"
        except Exception as e:
            print(f"Vision API error for {image_id}: {e}")
            return f"{manual_name} 图片 {image_id}"

    def _build_doc_id(self, item: Dict[str, Any], source_type: str) -> str:
        import uuid
        raw = json.dumps({"source_type": source_type, "item": item}, ensure_ascii=False, sort_keys=True)
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, raw))

    def _read_manual_file(self, file_path: Path) -> Tuple[str, List[str]]:
        content = file_path.read_text(encoding="utf-8")
        try:
            parsed = ast.literal_eval(content)
            if isinstance(parsed, list) and len(parsed) == 2 and isinstance(parsed[0], str) and isinstance(parsed[1], list):
                return parsed[0], [str(x) for x in parsed[1]]
        except Exception:
            pass
        return content, []

    def _chunk_text(self, text: str, max_chars: int = 700, overlap: int = 120) -> List[str]:
        normalized = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not normalized: return []
        paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
        chunks, current = [], ""
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
                    if piece.strip(): chunks.append(piece.strip())
                    if end >= len(para): break
                    start = max(0, end - overlap)
                current = ""
        if current: chunks.append(current)
        return chunks

    def _to_docs(self) -> Tuple[List[Document], List[str]]:
        docs, ids = [], []
        manual_files = sorted([p for p in self.docs_dir.glob("*.txt") if p.is_file()])

        for manual_file in manual_files:
            text, image_ids = self._read_manual_file(manual_file)
            manual_name = manual_file.stem
            clean_text = text.replace("<PIC>", " ")

            for chunk_idx, chunk in enumerate(self._chunk_text(clean_text)):
                item = {"manual": manual_name, "chunk_idx": chunk_idx, "content": chunk}
                docs.append(Document(page_content=chunk, metadata={"type": "text", "manual": manual_name}))
                ids.append(self._build_doc_id(item, "text"))

            for img_idx, image_id in enumerate(image_ids):
                image_path = self.image_name_to_path.get(image_id.lower(), "")
                image_summary = self._summarize_image_for_match(image_path, image_id, manual_name, text)
                item = {"manual": manual_name, "image_idx": img_idx, "image_id": image_id, "content": image_summary}
                docs.append(Document(page_content=image_summary, metadata={
                    "type": "image_summary", "manual": manual_name, "image_id": image_id, "image_path": image_path
                }))
                ids.append(self._build_doc_id(item, "image_summary"))
        return docs, ids

    def build_knowledge_base(self) -> Dict[str, int]:
        docs, ids = self._to_docs()
        if not docs: return {"docs": 0, "added": 0}
        self.local_docs = docs
        if self.vector_store is None: return {"docs": len(docs), "added": len(docs)}

        try:
            self.vector_store.add_documents(docs, ids=ids)
            added_count = len(docs)
        except Exception as e:
            print(f"Error adding docs: {e}")
            added_count = 0
        return {"docs": len(docs), "added": added_count}

    def search(self, query: str, top_k: int = 8) -> Dict[str, Any]:
        """用于回答产品说明书、使用方法、部件说明、故障排查、操作步骤等技术问题时，从向量库检索说明书信息。"""
        keyword_to_manual = {
            "洗碗机": "洗碗机手册", "空调": "空调手册", "微波炉": "微波炉手册", "冰箱": "冰箱手册",
            "烤箱": "烤箱手册", "净化器": "空气净化器手册", "相机": "相机手册", "发电机": "发电机手册",
            "电钻": "电钻手册", "吹风机": "吹风机手册", "水泵": "水泵手册", "VR": "VR头显手册",
            "键盘": "功能键盘手册", "摩托艇": "摩托艇手册", "温控器": "可编程温控器手册",
            "人体工学椅": "人体工学椅手册", "健身单车": "健身单车手册", "追踪器": "健身追踪器手册",
            "摩托车": "儿童电动摩托车手册", "鼠标": "蓝牙激光鼠标手册", "蒸汽清洁机": "蒸汽清洁机手册"
        }

        target_manuals = [m for k, m in keyword_to_manual.items() if k in query]

        from qdrant_client.http import models as rest
        any_type_conditions = [
            rest.FieldCondition(key="metadata.type", match=rest.MatchAny(any=["text", "image_summary"]))
        ]

        if target_manuals:
            any_type_conditions.append(
                rest.FieldCondition(key="metadata.manual", match=rest.MatchAny(any=target_manuals))
            )

        final_filter = rest.Filter(must=any_type_conditions)

        try:
            retrieved_docs = self.vector_store.similarity_search(
                query=query, k=top_k * 2, filter=final_filter
            )
        except Exception:
            retrieved_docs = self.vector_store.similarity_search(query=query, k=top_k * 2)

        context_lines = []
        seen_image_ids = set()
        images = []
        paths = []

        for i, doc in enumerate(retrieved_docs, start=1):
            m = doc.metadata
            doc_type = m.get("type", "text")
            manual = m.get("manual", "未知手册")

            if doc_type == "text":
                context_lines.append(f"[参考{i}][文本][{manual}] {doc.page_content.strip()}")
            elif doc_type == "image_summary":
                img_id = m.get("image_id")
                img_path = m.get("image_path", "")
                context_lines.append(f"[参考{i}][图片总结][{manual}] 图片ID: {img_id}。内容: {doc.page_content.strip()}")

                if img_id and img_id not in seen_image_ids:
                    seen_image_ids.add(img_id)
                    images.append(img_id)
                    paths.append(img_path)

        if not context_lines:
            text_res = "未在说明书中检索到相关内容。"
        else:
            text_res = "说明书检索结果如下：\n" + "\n".join(context_lines)

        return {
            "text": text_res,
            "image_list": images,
            "image_paths": paths,
            "references": context_lines
        }

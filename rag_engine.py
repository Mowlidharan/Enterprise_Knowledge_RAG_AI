"""
RAG Engine Service

Production-grade Retrieval-Augmented Generation engine
using ChromaDB and LlamaIndex.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    VectorStoreQueryMode,
)
from llama_index.vector_stores.chroma import (
    ChromaVectorStore,
)

from chunking_engine import TextChunker
from document_processor import DocumentProcessor

from llm_infrastructure_service import (
    embed_model,
    standalone_query_response,
    llm_hit,
    rag_config,
    reranker_llm,
    cLog,
)

from docsIntegrationPromptManager import (
    DEFAULT_CHAT_HISTORY_PROMPT,
    DEFAULT_QA_PROMPT,
)


# ============================================================
# Prompt Templates
# ============================================================

QUESTION_ANSWER_PROMPT = PromptTemplate(
    DEFAULT_QA_PROMPT,
    prompt_type=PromptType.QUESTION_ANSWER,
)

CHAT_HISTORY_PROMPT = PromptTemplate(
    DEFAULT_CHAT_HISTORY_PROMPT,
    prompt_type=PromptType.QUESTION_ANSWER,
)


# ============================================================
# Exceptions
# ============================================================

class RAGEngineError(Exception):
    """Base RAG engine exception."""


class DocumentAlreadyExistsError(RAGEngineError):
    """Raised when duplicate document exists."""


class CollectionNotFoundError(RAGEngineError):
    """Raised when collection is missing."""


# ============================================================
# Utility Functions
# ============================================================

SOURCE_LINK_PATTERN = re.compile(
    r"(?:Source\s*:?\s*)(https?://[^\s]+)",
    flags=re.IGNORECASE,
)


async def extract_source_link(
    text: str,
) -> Optional[str]:
    """
    Extract source URL from document text.
    """

    def _extract() -> Optional[str]:
        match = SOURCE_LINK_PATTERN.search(text)
        return match.group(1) if match else None

    return await asyncio.to_thread(_extract)


async def generate_content_hash(
    text: str,
) -> str:
    """
    Generate SHA256 hash for text content.
    """

    return await asyncio.to_thread(
        lambda: hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()
    )


# ============================================================
# RAG Engine
# ============================================================

class RagEngine:
    """
    Main Retrieval-Augmented Generation engine.
    """

    def __init__(self) -> None:
        """
        Initialize vector database and embedding model.
        """

        self.embedding_model = embed_model

        Settings.embed_model = self.embedding_model

        self.vector_database = (
            chromadb.PersistentClient(
                path=rag_config["chromaDbPath"]
            )
        )

        self.text_chunker = TextChunker(
            max_chunk_size=rag_config["ChunkLength"]
        )

    # ========================================================
    # Collection Management
    # ========================================================

    async def load_vector_index(
        self,
        collection_name: str,
        ingestion_nodes: Optional[
            List[TextNode]
        ] = None,
    ) -> VectorStoreIndex | str:
        """
        Load or create vector index.
        """

        collection = await asyncio.to_thread(
            self.vector_database.get_or_create_collection,
            collection_name,
        )

        vector_store = ChromaVectorStore(
            chroma_collection=collection
        )

        storage_context = (
            StorageContext.from_defaults(
                vector_store=vector_store
            )
        )

        if ingestion_nodes:

            embedded_nodes = await asyncio.gather(
                *[
                    self._embed_node(node)
                    for node in ingestion_nodes
                ]
            )

            await asyncio.to_thread(
                vector_store.add,
                embedded_nodes,
            )

            return "ingested"

        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=self.embedding_model,
        )

    async def _embed_node(
        self,
        node: TextNode,
    ) -> TextNode:
        """
        Generate embedding for node.
        """

        node.embedding = await asyncio.to_thread(
            self.embedding_model.get_text_embedding,
            node.text,
        )

        return node

    # ========================================================
    # Document Validation
    # ========================================================

    async def document_exists(
        self,
        collection_name: str,
        file_name: str,
    ) -> bool:
        """
        Check whether document already exists.
        """

        collections = {
            collection.name
            for collection
            in self.vector_database.list_collections()
        }

        if collection_name not in collections:
            return False

        collection = (
            self.vector_database.get_collection(
                collection_name
            )
        )

        result = collection.get(
            where={"file_name": file_name}
        )

        return bool(result.get("ids"))

    async def content_exists(
        self,
        collection_name: str,
        content_hash: str,
    ) -> bool:
        """
        Check whether document hash exists.
        """

        collection = (
            self.vector_database
            .get_or_create_collection(
                collection_name
            )
        )

        result = collection.get(ids=content_hash)

        return bool(result["ids"])

    # ========================================================
    # Metadata Storage
    # ========================================================

    async def insert_document_metadata(
        self,
        collection_name: str,
        file_path: str,
        file_name: str,
        description: str,
        content_hash: str,
        uploaded_by: str,
    ) -> Dict[str, Any]:
        """
        Store document metadata.
        """

        metadata_collection = (
            self.vector_database
            .get_or_create_collection(
                name=collection_name
            )
        )

        metadata_collection.add(
            ids=content_hash,
            documents=description,
            metadatas={
                "file_name": file_name,
                "file_path": file_path,
                "uploaded_by": uploaded_by,
                "uploaded_at": datetime.utcnow()
                .isoformat(),
            },
        )

        return {
            "status": True,
            "message": (
                "Metadata inserted successfully."
            ),
        }

    # ========================================================
    # Document Ingestion
    # ========================================================

    async def ingest_document(
        self,
        collection_name: str,
        file_path: str,
        user_id: str,
        spaceid: str,
    ) -> Dict[str, Any]:
        """
        Process and ingest document into vector store.
        """

        file_name = Path(file_path).name

        metadata_collection = (
            f"{collection_name}_metadata"
        )

        if await self.document_exists(
            metadata_collection,
            file_name,
        ):
            raise DocumentAlreadyExistsError(
                "Document already exists."
            )

        document_text = (
            await DocumentProcessor.extract_markdown(
                file_path
            )
        )

        content_hash = (
            await generate_content_hash(
                document_text
            )
        )

        (
            duplicate_content,
            source_link,
            text_chunks,
        ) = await asyncio.gather(
            self.content_exists(
                metadata_collection,
                content_hash,
            ),
            extract_source_link(document_text),
            self.text_chunker.split_text(
                document_text
            ),
        )

        if duplicate_content:
            error_msg = f"Duplicate document content for file: {file_name}"
            cLog(spaceid=spaceid, fileName="DocumentIngestion", message=f"ERROR: {error_msg}")
            raise DocumentAlreadyExistsError(error_msg)

        cLog(
            spaceid=spaceid,
            fileName="DocumentIngestion",
            message=f"Document split into {len(text_chunks)} chunks for file: {file_name}"
        )

        nodes = [
            TextNode(
                text=chunk,
                metadata={
                    "file_name": file_name,
                    "uploaded_by": user_id,
                    "source_link": source_link,
                    "file_path": file_path,
                },
            )
            for chunk in text_chunks
        ]

        cLog(
            spaceid=spaceid,
            fileName="DocumentIngestion",
            message=f"Starting ingestion for file: {file_name} in collection: {collection_name} by User: {user_id}"
        )

        await self.load_vector_index(
            collection_name=collection_name,
            ingestion_nodes=nodes,
        )

        document_summary = (
            await self.generate_response(
                question=(
                    f"Summarize {file_name}"
                ),
                collection_name=collection_name,
                chat_history=[],
                spaceid=spaceid,
            )
        )

        await self.insert_document_metadata(
            collection_name=metadata_collection,
            file_path=file_path,
            file_name=file_name,
            description=document_summary,
            content_hash=content_hash,
            uploaded_by=user_id,
        )

        cLog(
            spaceid=spaceid,
            fileName="DocumentIngestion",
            message=f"Successfully ingested file: {file_name}"
        )

        return {
            "status": True,
            "message": (
                "Document ingested successfully."
            ),
        }

    # ========================================================
    # Response Generation
    # ========================================================

    async def generate_response(
        self,
        question: str,
        collection_name: str,
        chat_history: List[Dict[str, str]],
        spaceid: str,
        return_sources: bool = False,
    ) -> str | Dict[str, Any]:
        """
        Generate RAG response.
        """

        if chat_history:
            reframed_query_prompt = (
                CHAT_HISTORY_PROMPT.format(
                    input_str=question,
                    chat_history=chat_history,
                )
            )

            reframed_query = (
                await standalone_query_response(
                    prompt=reframed_query_prompt
                )
            )
        else:
            # No prior turns to resolve references against — reframing has
            # nothing to do, and asking a small local LLM to do it anyway
            # risks it "fixing" an already-standalone question into garbage.
            reframed_query = question

        index = await self.load_vector_index(
            collection_name=collection_name,
        )

        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=rag_config[
                "SimilarityTopK"
            ],
            vector_store_query_mode=(
                VectorStoreQueryMode.DEFAULT
            ),
            embed_model=self.embedding_model,
        )

        retrieved_nodes = (
            await retriever.aretrieve(
                reframed_query
            )
        )

        cLog(
            spaceid=spaceid,
            fileName="RAGResponse",
            message=f"Retrieved {len(retrieved_nodes)} raw nodes from vector store"
        )


        context_segments = []
        source_records = []
        for node in retrieved_nodes:
            if node.get_score() > rag_config["NodeScore"]:
                context_segments.append(node.get_content())
                source_records.append({
                    "file_name": node.metadata.get("file_name", "Unknown document"),
                    "source_link": node.metadata.get("source_link"),
                    "file_path": node.metadata.get("file_path"),
                    "score": round(node.get_score(), 3),
                })

        cLog(
            spaceid=spaceid,
            fileName="RAGResponse",
            message=f"Filtered to {len(context_segments)} relevant context segments (Score > {rag_config['NodeScore']})"
        )

        if not context_segments:
            no_context_msg = rag_config.get("DEFAULT_NO_CONTEXT_MESSAGE", "I don't have enough information to answer this.")
            if return_sources:
                return {
                    "answer": no_context_msg,
                    "sources": []
                }
            return no_context_msg

        context_text = "\n".join(
            context_segments
        )

        final_prompt = (
            QUESTION_ANSWER_PROMPT.format(
                query_str=reframed_query,
                context_str=context_text,
            )
        )

        cLog(
            spaceid=spaceid,
            fileName="RAGResponse",
            message=f"Generating response for question in collection: {collection_name}"
        )

        response = await llm_hit(
            primary_prompt=final_prompt,
            ollama_prompt=final_prompt,
            primary_llm="ollama",
            secondary_llm="ollama",
            response_type="text",
            node_id="rag-response",
            userid="system",
            spaceid=spaceid,
            ollama_model_name=rag_config[
                "ollamaModel"
            ],
        )

        cLog(
            spaceid=spaceid,
            fileName="RAGResponse",
            message="Response generated successfully"
        )
        
        if return_sources:
            return {
                "answer": response,
                "sources": source_records
            }

        return response

    async def get_document_count(self, collection_name: str) -> int:
        """Get number of unique documents in a collection."""
        try:
            collection = self.vector_database.get_collection(f"{collection_name}_metadata")
            results = collection.get()
            return len(set(results.get("ids", [])))
        except Exception:
            return 0


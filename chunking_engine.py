"""
Text Chunking Service

Provides asynchronous sentence-aware text chunking
for RAG pipelines and document processing systems.
"""

from __future__ import annotations

import asyncio
import re
from typing import List


# ============================================================
# Constants
# ============================================================

WHITESPACE_PATTERN = re.compile(r"\s+")
SENTENCE_SPLIT_PATTERN = re.compile(
    r"(?<=[.!?])\s+"
)


# ============================================================
# Exceptions
# ============================================================

class TextChunkingError(Exception):
    """Raised when text chunking fails."""


# ============================================================
# Text Chunker
# ============================================================

class TextChunker:
    """
    Sentence-aware text chunking utility.
    """

    def __init__(
        self,
        max_chunk_size: int = 1000,
    ) -> None:
        """
        Initialize text chunker.

        Args:
            max_chunk_size:
                Maximum characters allowed per chunk.
        """

        if max_chunk_size <= 0:
            raise ValueError(
                "max_chunk_size must be greater than 0"
            )

        self.max_chunk_size = max_chunk_size

    async def split_text(
        self,
        text: str,
    ) -> List[str]:
        """
        Split text into sentence-aware chunks.

        Args:
            text:
                Input text content.

        Returns:
            List of text chunks.

        Raises:
            TextChunkingError:
                If chunking operation fails.
        """

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        if not text.strip():
            return []

        try:
            return await asyncio.to_thread(
                self._split_text_sync,
                text,
            )

        except Exception as exc:
            raise TextChunkingError(
                "Failed to split text into chunks"
            ) from exc

    def _split_text_sync(
        self,
        text: str,
    ) -> List[str]:
        """
        Synchronous text chunking worker.

        Args:
            text:
                Raw input text.

        Returns:
            List of chunked text segments.
        """

        normalized_text = self._normalize_text(text)

        sentences = SENTENCE_SPLIT_PATTERN.split(
            normalized_text
        )

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            sentence_length = len(sentence)

            # Handle oversized sentence
            if sentence_length > self.max_chunk_size:

                if current_chunk:
                    chunks.append(
                        " ".join(current_chunk).strip()
                    )
                    current_chunk = []
                    current_length = 0

                oversized_chunks = (
                    self._split_large_sentence(
                        sentence
                    )
                )

                chunks.extend(oversized_chunks)

                continue

            projected_length = (
                current_length
                + sentence_length
                + 1
            )

            if projected_length > self.max_chunk_size:

                chunks.append(
                    " ".join(current_chunk).strip()
                )

                current_chunk = [sentence]
                current_length = sentence_length

            else:
                current_chunk.append(sentence)
                current_length = projected_length

        if current_chunk:
            chunks.append(
                " ".join(current_chunk).strip()
            )

        return chunks

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        """
        Normalize whitespace in text.

        Args:
            text:
                Raw text input.

        Returns:
            Cleaned text.
        """

        return WHITESPACE_PATTERN.sub(
            " ",
            text.strip(),
        )

    def _split_large_sentence(
        self,
        sentence: str,
    ) -> List[str]:
        """
        Split oversized sentence into smaller chunks.

        Args:
            sentence:
                Sentence exceeding chunk limit.

        Returns:
            List of smaller chunks.
        """

        return [
            sentence[i : i + self.max_chunk_size]
            for i in range(
                0,
                len(sentence),
                self.max_chunk_size,
            )
        ]


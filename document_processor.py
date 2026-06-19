"""
Document Processing Service

Provides asynchronous document parsing and markdown extraction
using Docling with production-grade error handling.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from docling.document_converter import DocumentConverter


# ============================================================
# Constants
# ============================================================

MARKDOWN_CLEANUP_PATTERN = re.compile(
    r"(&gt;&gt;|<!--\s*image\s*-->)",
    flags=re.IGNORECASE,
)


# ============================================================
# Document Processing
# ============================================================

class DocumentProcessingError(Exception):
    """Raised when document processing fails."""


class DocumentProcessor:
    """
    Handles document parsing and markdown extraction.
    """

    @staticmethod
    async def extract_markdown(
        file_path: str | Path,
    ) -> str:
        """
        Extract cleaned markdown content from a document.

        Args:
            file_path:
                Path to the input document.

        Returns:
            Cleaned markdown text.

        Raises:
            FileNotFoundError:
                If the document does not exist.

            DocumentProcessingError:
                If parsing or extraction fails.
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Document not found: {path}"
            )

        if not path.is_file():
            raise DocumentProcessingError(
                f"Invalid file path: {path}"
            )

        try:
            converter = DocumentConverter()
            return await asyncio.to_thread(
                DocumentProcessor._process_document,
                converter,
                path,
            )

        except Exception as exc:
            raise DocumentProcessingError(
                f"Failed to process document: {path.name}"
            ) from exc

    @staticmethod
    def _process_document(
        converter: DocumentConverter,
        file_path: Path,
    ) -> str:
        """
        Synchronous document parsing worker.

        Args:
            file_path:
                Path to document.

        Returns:
            Cleaned markdown content.
        """

        conversion_result = converter.convert(
            str(file_path)
        )

        markdown_content = (
            conversion_result.document.export_to_markdown()
        )

        cleaned_markdown = MARKDOWN_CLEANUP_PATTERN.sub(
            "",
            markdown_content,
        )

        return cleaned_markdown.strip()


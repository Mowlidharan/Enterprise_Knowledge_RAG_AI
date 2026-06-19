
"""
Document RAG Service API

Production-ready FastAPI service for:
- Document uploads
- ZIP extraction
- RAG-based querying
- Conversation persistence
- Structured logging
"""

import asyncio
import os
import re
import shutil
import traceback
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from rag_engine import RagEngine
from llm_infrastructure_service import (
    cLog,
    llm_hit,
    rag_config,
    save_chat_record,
    _mkdir,
)
from docsIntegrationPromptManager import RESPONSE_REPHRASE_PROMPT


# ============================================================
# Application Configuration
# ============================================================

app = FastAPI(title="Document RAG API", version="1.0.0")

UPLOAD_DIRECTORY = Path(rag_config["UploadedDirFile"])
EXTRACTION_DIRECTORY = Path(rag_config["ExtractedDirFile"])
VALID_DOCUMENT_EXTENSIONS = set(rag_config["document_validate"])

FALLBACK_RESPONSE_PHRASES = {
    "i don't have enough",
    "i dont have enough",
}

_mkdir(str(UPLOAD_DIRECTORY))
_mkdir(str(EXTRACTION_DIRECTORY))


# ============================================================
# Utility Functions
# ============================================================

async def save_uploaded_file(
    upload_file: UploadFile,
    destination_dir: Path,
) -> Path:
    """
    Save an uploaded file to disk.

    Args:
        upload_file: Uploaded file object.
        destination_dir: Directory where the file will be stored.

    Returns:
        Path to the stored file.
    """

    destination_path = destination_dir / upload_file.filename

    with destination_path.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return destination_path


async def extract_zip_file(
    zip_file_path: Path,
    output_directory: Path,
) -> List[Path]:
    """
    Extract ZIP archive and return supported document files.

    Args:
        zip_file_path: ZIP archive path.
        output_directory: Extraction directory.

    Returns:
        List of extracted valid document paths.
    """

    extracted_files: List[Path] = []

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(output_directory)

    for root, _, files in os.walk(output_directory):
        for file_name in files:
            extension = file_name.split(".")[-1].lower()

            if extension in VALID_DOCUMENT_EXTENSIONS:
                extracted_files.append(Path(root) / file_name)

    return extracted_files


def normalize_response_text(text: str) -> str:
    """
    Format response text with proper newlines.

    Args:
        text: Raw response text.

    Returns:
        Formatted response text.
    """

    return re.sub(r"\.(?!\s*\n)\s*", ".\n", text)


async def ingest_document(
    file_path: Path,
    tenant_id: str,
    business_id: str,
    user_id: str,
    workspace_id: str,
    collection_name: str,
) -> Dict[str, Any]:
    """
    Ingest a single document into the RAG system.

    Returns:
        Ingestion status payload.
    """

    try:
        ingestion_response = await rag_engine.ingest_document(
            collection_name=collection_name,
            file_path=str(file_path),
            user_id=user_id,
            spaceid=workspace_id,
        )

        return {
            "file": file_path.name,
            "status": ingestion_response["status"],
            "message": ingestion_response["message"],
        }

    except Exception as exc:
        return {
            "file": file_path.name,
            "status": False,
            "message": str(exc),
        }


# ============================================================
# Engine Initialization
# ============================================================

rag_engine = RagEngine()


# ============================================================
# Upload API
# ============================================================

@app.post("/api/v1/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    collection_name: str = Form(...),
    tenant_id: str = Form(...),
    business_id: str = Form(...),
    workspace_id: str = Form(...),
    user_id: str = Form(...),
):
    """
    Upload and ingest documents into the RAG system.

    Supported:
    - PDF
    - DOCX
    - Images
    - ZIP archives
    """

    try:
        if not files:
            raise HTTPException(
                status_code=400,
                detail="No files uploaded.",
            )

        ingested_files = []
        failed_files = []
        skipped_files = []
        ingestion_results = []

        stored_files: List[Path] = []

        cLog(
            spaceid=workspace_id,
            fileName="DocumentUploadAPI",
            message=f"UserID: {user_id} | Received {len(files)} file(s)",
        )

        for upload in files:
            if not upload.filename:
                continue

            stored_path = await save_uploaded_file(
                upload,
                UPLOAD_DIRECTORY,
            )

            stored_files.append(stored_path)

        for file_path in stored_files:
            extension = file_path.suffix.lower().replace(".", "")

            try:
                if extension == "zip":

                    extracted_files = await extract_zip_file(
                        file_path,
                        EXTRACTION_DIRECTORY,
                    )

                    if file_path.exists():
                        file_path.unlink()

                    if not extracted_files:
                        skipped_files.append(file_path.name)
                        cLog(spaceid=workspace_id, fileName="DocumentUploadAPI", message=f"No valid documents found in ZIP: {file_path.name}")
                        continue

                    cLog(spaceid=workspace_id, fileName="DocumentUploadAPI", message=f"Extracted {len(extracted_files)} files from ZIP: {file_path.name}")

                    ingestion_tasks = [
                        ingest_document(
                            extracted_file,
                            tenant_id,
                            business_id,
                            user_id,
                            workspace_id,
                            collection_name,
                        )
                        for extracted_file in extracted_files
                    ]

                    task_results = await asyncio.gather(
                        *ingestion_tasks
                    )

                    for result in task_results:
                        ingestion_results.append(result)

                        if result["status"]:
                            ingested_files.append(result["file"])
                        else:
                            failed_files.append(result["file"])

                elif extension in VALID_DOCUMENT_EXTENSIONS:

                    result = await ingest_document(
                        file_path=file_path,
                        tenant_id=tenant_id,
                        business_id=business_id,
                        user_id=user_id,
                        workspace_id=workspace_id,
                        collection_name=collection_name,
                    )

                    ingestion_results.append(result)

                    if result["status"]:
                        ingested_files.append(result["file"])
                    else:
                        failed_files.append(result["file"])

                else:
                    skipped_files.append(file_path.name)

            except Exception as exc:
                error_msg = f"Failed to process {file_path.name}: {str(exc)}"
                cLog(spaceid=workspace_id, fileName="DocumentUploadAPI", message=f"ERROR: {error_msg}")
                failed_files.append(error_msg)

        summary = {
            "ingested": ingested_files,
            "failed": failed_files,
            "skipped": skipped_files,
            "details": ingestion_results,
        }

        return JSONResponse(
            status_code=200,
            content={
                "status": True,
                "message": "Document ingestion completed successfully.",
                "summary": summary,
            },
        )

    except HTTPException:
        raise

    except Exception as exc:
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "status": False,
                "message": str(exc),
            },
        )


# ============================================================
# Query API
# ============================================================

@app.post("/api/v1/documents/query")
async def query_documents(request: Request):
    """
    Query documents using Retrieval-Augmented Generation (RAG).
    """

    request_data = await request.json()

    question = request_data.get("prompt")
    collection_name = request_data.get("collection_name")

    tenant_id = request_data.get("t1")
    business_id = request_data.get("b1")
    workspace_id = request_data.get("spaceid")
    user_id = request_data.get("userid")

    current_chat_id = request_data.get("currentchatid")
    next_chat_id = request_data.get("nextchatid")
    ui_key = request_data.get("ui_key", False)

    try:
        required_fields = {
            "prompt": question,
            "collection_name": collection_name,
            "tenant_id": tenant_id,
            "business_id": business_id,
            "workspace_id": workspace_id,
        }

        missing_fields = [
            field_name
            for field_name, field_value in required_fields.items()
            if field_value in (None, "", [])
        ]

        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Missing required field(s): "
                    + ", ".join(missing_fields)
                ),
            )

        chat_state = {
            "t1": tenant_id,
            "b1": business_id,
            "spaceid": workspace_id,
            "userid": user_id,
            "conversationid": current_chat_id,
            "chat_conversation": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
        }

        save_chat_record(
            t1b1=f"{tenant_id}_{business_id}",
            spaceid=workspace_id,
            state=chat_state,
            status="0",
            ui_key=False,
        )

        rag_response_msg = await rag_engine.generate_response(
            question=question,
            collection_name=collection_name,
            chat_history=[],
            spaceid=workspace_id,
        )

        rag_response = {"status": True, "message": rag_response_msg}

        if any(
            phrase in rag_response["message"].lower()
            for phrase in FALLBACK_RESPONSE_PHRASES
        ):
            final_response = (
                "I could not confidently find the answer. "
                "Please rephrase your question and try again."
            )

        else:
            rephrase_prompt = RESPONSE_REPHRASE_PROMPT.format(
                user_query=question,
                response=rag_response["message"],
            )

            final_response = await llm_hit(
                primary_prompt=rephrase_prompt,
                ollama_prompt=rephrase_prompt,
                primary_llm="ollama",
                secondary_llm="ollama",
                userid=user_id,
                spaceid=workspace_id,
                ollama_model_name=(
                    rag_config["responseOllamaModel"]
                ),
                response_type="text",
                node_id="document-agent",
            )

        response_payload = {
            "status": True,
            "conversation_id": next_chat_id,
            "message": normalize_response_text(
                final_response
            ),
            "data_representation": 0,
        }

        return JSONResponse(response_payload)

    except HTTPException:
        raise

    except Exception as exc:
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "status": False,
                "message": str(exc),
                "conversation_id": next_chat_id,
            },
        )


# ============================================================
# Application Entry Point
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=1232,
        reload=True,
    )

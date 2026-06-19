"""
LLM Infrastructure Service

Production-grade utility layer for:
- Configuration loading
- HTTP communication
- LLM routing
- Embedding generation
- Token tracking
- Reranking
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import httpx
import pandas as pd
import pyarrow.feather as feather
from pydantic import Field

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.postprocessor import LLMRerank
from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding,
)


# ============================================================
# Configuration
# ============================================================

CONFIG_FILE_PATH = Path(
    r"D:\Mowlidharan\RAG\docsagentconfig.json"
)


def load_configuration(
    config_path: Path,
) -> Dict[str, Any]:
    """
    Load application configuration.
    """

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: "
            f"{config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        return json.load(config_file)


application_config = load_configuration(
    CONFIG_FILE_PATH
)


# ============================================================
# Constants
# ============================================================

UNCERTAIN_RESPONSE_PHRASES = {
    "i don't have enough",
    "i dont have enough",
}


# ============================================================
# HTTP Client
# ============================================================

class AsyncHTTPClient:
    """
    Reusable async HTTP client service.
    """

    DEFAULT_TIMEOUT = 60.0

    @staticmethod
    async def post_json(
        url: str,
        payload: Dict[str, Any],
        access_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send async POST request.
        """

        headers = {
            "Content-Type": "application/json"
        }

        if access_token:
            headers["Authorization"] = (
                f"Bearer {access_token}"
            )

        async with httpx.AsyncClient(
            timeout=AsyncHTTPClient.DEFAULT_TIMEOUT
        ) as client:

            response = await client.post(
                url=url,
                headers=headers,
                json=payload,
            )

            response.raise_for_status()

            return response.json()


# ============================================================
# Token Tracking
# ============================================================

class TokenTrackingService:
    """
    Tracks LLM token usage.
    """

    @staticmethod
    def persist_token_usage(
        response_payload: Dict[str, Any],
        request_metadata: Dict[str, Any],
    ) -> None:
        """
        Store token usage metadata.
        """

        token_directory = (
            Path(
                application_config[
                    "TokenCalculationFilePath"
                ]
            )
            / "metadata"
            / "tokenDetails"
        )

        token_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        token_file = (
            token_directory
            / (
                f"{request_metadata['spaceid']}"
                "_tokendetails"
            )
        )

        usage_record = {
            "spaceid": request_metadata[
                "spaceid"
            ],
            "nodeid": request_metadata[
                "nodeid"
            ],
            "userid": request_metadata[
                "userid"
            ],
            "input_tokens": (
                response_payload["data"]
                .get("input_tokens", 0)
            ),
            "output_tokens": (
                response_payload["data"]
                .get("output_tokens", 0)
            ),
            "total_tokens": (
                response_payload["data"]
                .get("total_tokens", 0)
            ),
            "request_id": (
                response_payload["data"]
                .get("request_id", "")
            ),
        }

        dataframe = pd.DataFrame(
            [usage_record]
        )

        if token_file.exists():

            existing_dataframe = (
                feather.read_feather(
                    token_file
                )
            )

            dataframe = pd.concat(
                [
                    existing_dataframe,
                    dataframe,
                ],
                ignore_index=True,
            )

        feather.write_feather(
            dataframe,
            token_file,
            compression="zstd",
        )


# ============================================================
# Ollama Client
# ============================================================

class OllamaClient:
    """
    Ollama LLM client.
    """

    @staticmethod
    async def generate_chat(
        prompt: str,
        model_name: str,
    ) -> Dict[str, Any]:
        """
        Generate LLM response from Ollama Chat API.
        """

        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }

        print(f"DEBUG: Calling Ollama at {application_config['ollamaChatURL']}")
        print(f"DEBUG: Payload: {json.dumps(payload, indent=2)}")
        
        return await AsyncHTTPClient.post_json(
            url=application_config["ollamaChatURL"],
            payload=payload,
        )


# Gemini Client removal


# ============================================================
# LLM Router
# ============================================================

class LLMRoutingService:
    """
    Handles LLM routing and fallback logic.
    """

    @staticmethod
    async def generate_response(
        primary_provider: str,
        secondary_provider: str,
        primary_prompt: str,
        fallback_prompt: str,
        response_type: Literal[
            "json",
            "text",
        ] = "text",
        model_name: Optional[str] = None,
        request_metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Union[Dict[str, Any], str]:
        """
        Generate response with fallback support.
        """

        try:

            primary_response = (
                await LLMRoutingService
                ._execute_provider(
                    provider=primary_provider,
                    prompt=primary_prompt,
                    response_type=response_type,
                    model_name=model_name,
                    request_metadata=(
                        request_metadata
                    ),
                )
            )

            return primary_response

        except Exception as exc:
            print(f"LLM execution failed: {exc}")
            raise

    @staticmethod
    async def _execute_provider(
        provider: str,
        prompt: str,
        response_type: str,
        model_name: Optional[str] = None,
        request_metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Union[Dict[str, Any], str]:
        """
        Execute provider request.
        """

        if provider.lower() == "ollama":

            response = await OllamaClient.generate_chat(
                prompt=prompt,
                model_name=model_name or application_config["ollamaModel"],
            )

            if request_metadata:
                # Mock token calculation as Ollama response structure for chat is different
                # message: {role: assistant, content: ...}
                TokenTrackingService.persist_token_usage(
                    {
                        "data": {
                            "input_tokens": len(prompt) // 4,
                            "output_tokens": len(response["message"]["content"]) // 4,
                            "total_tokens": (len(prompt) + len(response["message"]["content"])) // 4,
                        }
                    },
                    request_metadata,
                )

            raw_output = response["message"]["content"]

        else:
             raise ValueError(f"Unsupported provider: {provider}")

        if response_type == "json":
            return json.loads(
                raw_output
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        return raw_output


# ============================================================
# Reranker Service
# ============================================================

class RerankerService:
    """
    LLM reranker factory.
    """

    @staticmethod
    def create_reranker(
        provider: str,
    ) -> LLMRerank:
        """
        Create reranker instance.
        """

        # For now, we only support local reranking or a generic one.
        # If Gemini is removed, we might need a different reranker.
        # Using a simpler postprocessor if no provider is matched.
        
        raise NotImplementedError("Reranker currently disabled as it depended on Gemini.")


# ============================================================
# Embedding Service
# ============================================================

class CustomEmbeddingService(
    BaseEmbedding
):
    """
    External embedding service wrapper.
    """

    api_url: str = Field(...)
    framework_type: str = Field(...)
    model_name: str = Field(...)

    async def _request_embedding(
        self,
        text: str,
    ) -> List[float]:

        payload = {
            "model": self.model_name,
            "prompt": text,
        }

        response = await AsyncHTTPClient.post_json(
            url=self.api_url,
            payload=payload,
        )

        return response["embedding"]

    async def _aget_query_embedding(
        self,
        query: str,
    ):
        return await self._request_embedding(
            query
        )

    async def _aget_text_embedding(
        self,
        text: str,
    ):
        return await self._request_embedding(
            text
        )

    def _get_query_embedding(
        self,
        query: str,
    ):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return loop.create_task(self._request_embedding(query))
            return loop.run_until_complete(self._request_embedding(query))
        except RuntimeError:
            return asyncio.run(self._request_embedding(query))

    def _get_text_embedding(
        self,
        text: str,
    ):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return loop.create_task(self._request_embedding(text))
            return loop.run_until_complete(self._request_embedding(text))
        except RuntimeError:
            return asyncio.run(self._request_embedding(text))


# ============================================================
# Shared Embedding Model
# ============================================================

embedding_model = (
    CustomEmbeddingService(
        api_url=application_config[
            "ollamaEmbdURL"
        ],
        framework_type=application_config[
            "ollama_framework"
        ],
        model_name=application_config[
            "ollamaEmbdModel"
        ],
    )
)

embed_model = embedding_model


# ============================================================
# Helper Functions
# ============================================================

def _mkdir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def cLog(spaceid, fileName, message):
    """
    Log a message to a space-specific log file.
    Only spaceid, fileName, and message are used for easier tracking.
    """
    log_dir = Path(application_config["logPath"]) / spaceid
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{fileName}.log"
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [{spaceid}] [{fileName}] {message}\n")


def save_chat_record(t1b1, spaceid, state, status, ui_key):
    chat_dir = Path(application_config["ChatDir"]) / t1b1 / spaceid
    chat_dir.mkdir(parents=True, exist_ok=True)
    chat_file = chat_dir / f"{state['conversationid']}.json"
    with open(chat_file, "w") as f:
        json.dump(state, f)


async def llm_hit(
    primary_prompt,
    ollama_prompt,
    primary_llm,
    secondary_llm,
    userid,
    spaceid,
    ollama_model_name,
    response_type,
    node_id,
):
    metadata = {
        "spaceid": spaceid,
        "userid": userid,
        "nodeid": node_id
    }
    return await LLMRoutingService.generate_response(
        primary_provider="ollama",
        secondary_provider="ollama",
        primary_prompt=ollama_prompt,
        fallback_prompt=ollama_prompt,
        response_type=response_type,
        model_name=ollama_model_name,
        request_metadata=metadata
    )


async def standalone_query_response(prompt: str) -> str:
    # Generic replacement for standalone_query_response using Ollama
    metadata = {"spaceid": "internal", "userid": "system", "nodeid": "internal"}
    return await LLMRoutingService.generate_response(
        primary_provider="ollama",
        secondary_provider="ollama",
        primary_prompt=prompt,
        fallback_prompt=prompt,
        response_type="text",
        model_name=application_config["ollamaModel"],
        request_metadata=metadata
    )


def reranker_llm(llm_type=None):
    return RerankerService


rag_config = application_config


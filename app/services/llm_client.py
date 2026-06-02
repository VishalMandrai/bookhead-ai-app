# ─────────────────────────────────────────────────────────────────────────────
# app/services/llm_client.py
#
#
# WHY A SEPARATE CLIENT MODULE?
#   LLMRecommendationService and LLMCatalogService both need to call the
#   LLM API. Extracting the raw API mechanics here means:
#     - Both services share the same retry / error-handling logic
#     - The Anthropic client is instantiated once and reused
#     - Swapping to a different LLM provider requires changing only this file
#     - Tests can inject a MockClient without touching service code
#
# WEB SEARCH INTEGRATION:
#   The LLM API supports a built-in web_search tool. When enabled, the
#   model can call web search automatically during generation. We enable it for
#   both the reader (needs current book info) and librarian (needs to verify
#   titles/authors) tasks.
#
# RETRY STRATEGY (tenacity):
#   Heavy LLM's API like GPT-5, Anthropic Sonnet can return 429 (rate limit) or 529 (overloaded) 
#   transiently. We retry up to 3 times with exponential backoff (2s → 4s → 8s) before
#   raising an LLMError to the caller.
#
# SOLID (Single Responsibility):
#   This class only handles HTTP transport to LLM Service Provider. Prompt construction
#   lives in prompt_builder.py; response parsing lives in llm_parser.py.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Any

# from tenacity import (
#     retry,
#     retry_if_exception_type,
#     stop_after_attempt,
#     wait_exponential,
# )

from app.core.exceptions import LLMError
from app.core.logging import get_logger

logger = get_logger(__name__)


# ─────────────── For on-system Gemma LLM ───────────────────────────────────────────────────────
class GemmaLLMClient:
    """
    Instaniates Gemma LLM client object. Sets configuration.
    Make LLM call and return LLM response body.
    """

    def __init__(
        self,
        model_path: str,
        context_window_len: int,
        n_threads: int,
        batch_size: int,
        offload_n_gpu_layers: int,
        
        max_inf_tokens: int,
        llm_temperature: float,
    ) -> None:
        """
        Args:
        model_path:             Path to Gemma model in directory
        context_window_len:     Model Context Window
        n_threads:              Number of threads to be used for inference
        batch_size:             Number of input prompt tokens to be processed internally by model
        offload_n_gpu_layers:   Number of model layers to be offloaded to GPU
        
        max_inf_tokens:         Max tokens that model generates on inference
        llm_temperature:        Temperture param for Gemma LLM
        """
        self._model_path = model_path
        self._n_ctx = context_window_len
        self._n_threads = n_threads
        self._n_batch = batch_size
        self._offload_n_gpu_layers = offload_n_gpu_layers
        
        self._max_tokens = max_inf_tokens
        self._llm_temp = llm_temperature
        
        
        # load Gemma model on app startup
        self._model = None
        try:
            self.load_model()
        except:
            raise LLMError(
                        "Error loading Gemma model. Check Gemma configuration in config.py",
                    )

            
    def load_model(self) -> Any:
        """
        Instantiate the Gemma-2B Model.
        """
        if self._model is None:
            from llama_cpp import Llama
            
            llm = Llama(
                model_path=self._model_path ,        #Deafult Path: "models/gemma-2b-it.Q4_K_M.gguf"
                
                n_ctx=(self._n_ctx) ,       
                                # Context Window: maximum number of tokens the model can “remember” in 
                                # one inference session. 
                                # Costs: large CW, higher RAM usage, Slower inference, larger KV cache
                                # NOTE: n_ctx of 1024 tokens means llama.cpp on Llama() call reserves RAM 
                                # enough memory to store KV cache of 1024 tokens. So this param must be 
                                # optimised as per workload needs. Higher n_ctx, locks unnecessary RAM 
                                # memory; wasteful -> unused. 
                
                n_threads=(self._n_threads) ,
                                # Controls: number of CPU threads used during inference.
                                # Affects: prompt processing, token generation, matrix multiplications
                                # Too high is BAD - if n_threads=32 ; on an 8-core machine:
                                # Thread contention, Cache thrashing & Worse performance
                                
                n_batch=(self._n_batch) ,
                                # Batch Processing: how many prompt tokens are processed simultaneously 
                                # internally. Affects prompt ingestion speed.
                                # Larger n_batch - Benefits:
                                #                  Faster prompt evaluation & Better CPU vectorization
                                # Costs: more RAM usage
                        
                use_mmap=True ,    # Memory-map the model file instead of fully copying into RAM.
                                # Without mmap: Model - from disk → fully loaded into RAM
                                # Consumes: Startup time & Large memory allocation
                                # With mmap: OS lazily loads required model pages from disk.
                
                use_mlock=False ,  
                                # use_mlock: If True: OS tries to lock model pages into physical RAM
                                # Benefits: avoids swap latency & more stable inference latency
                                # Costs: requires enough RAM & may need admin permissions 
                
                n_gpu_layers=(self._offload_n_gpu_layers) ,
                                # n_gpu_layers -> Controls:how many transformer layers are offloaded to GPU.
                                # 0 Means: CPU-only inference; Everything runs on CPU.
                                # If GPU exists:
                                # Example: n_gpu_layers=20 -> Then: first 20 layers run on GPU
                                #                             remaining layers on CPU
                                # -1 Usually means: offload all possible layers to GPU

                f16_kv=True ,
                                # Controls precision of: KV cache storage.
                                # What is KV cache? Transformer stores: Attention keys & Attention values
                                #                                       for every previous token.
                                # This cache grows with: Context Window Length
                                # If True: Uses -> float16; instead of: float32
                                # Benefits: almost half KV cache RAM
                                #           faster memory access
                                #           Tiny quality impact.
                
                verbose=False  # Reduce verbose logging
                )
            
        self._model = llm
        logger.info("Gemma_llm_initialised", model=self._model_path)

        return self._model


    # ── API for Text Summarization ─────────────────────────────────────────────────────────

    def generate_summary(
        self,
        prompt: str,
    ) -> dict:
        """
        Send a prompt message to Gemma model for inference and return
        the assistant's complete response json for parsing.

        Args:
            prompt: Gemma formatted, complete text prompt describing assistant's 
                    role and its task. Specifies constraints and output schema.
        Returns:
            The assistant's response json.

        Raises:
            LLMError: If the inference call fails.
        """
        # Setting up inference params for text summarization:
        formatted_prompt = f"""
                    <start_of_turn>user
                    {prompt}<end_of_turn>
                    <start_of_turn>model
                    """
        
        logger.info(
            "Gemma_llm_call_start",
            model_path=self._model_path,
            prompt_length=len(prompt),
        )
        
        try:
            output = self._model(
                                formatted_prompt,
                                max_tokens= (self._max_tokens) ,
                                temperature=(self._llm_temp) ,
                                top_p=0.9,
                                repeat_penalty=1.1,
                                stop=["<end_of_turn>"]
                            )
        except Exception as exc:
            raise LLMError(
                f"Gemma inference call failed: {exc}",
                detail={"model": self._model_path},
            ) from exc

        logger.info(
            "llm_call_complete",
            model_path=self._model_path,
        )
        return output
        
        
# ──────────────────────────────────────────────────────────────────────────────────────────────


# ──────────── For Heavy LLM API Service ─────────────────────────────────────────────────────── 

class AnthropicLLMClient:
    """
    Thin wrapper around the Anthropic messages API.

    Handles client construction, web search tool setup, retry logic,
    and surfacing raw response content to callers.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 2048,
        enable_web_search: bool = True,
    ) -> None:
        """
        Args:
            api_key:           Anthropic API key from settings.
            model:             Model identifier (must support tool use + web search).
            max_tokens:        Maximum tokens in the completion.
            enable_web_search: Whether to include the web_search tool in calls.
        """
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._enable_web_search = enable_web_search

        # Lazy-loaded Anthropic client
        self._client: Any | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Send a system + user message pair to the Anthropic API and return
        the assistant's text response as a plain string.

        Web search is enabled when self._enable_web_search is True; the model
        may make multiple search calls internally before returning its final
        answer. We extract only the final text block from the response.

        Args:
            system_prompt: The system-role message (role / constraints).
            user_prompt:   The user-role message (task + data).

        Returns:
            The assistant's response text (may be JSON or prose).

        Raises:
            LLMError: If the API call fails after all retries, or if the
                      response contains no text content.
        """
        client = self._get_client()

        # Build the tools list — only web_search for now
        tools = []
        if self._enable_web_search:
            tools.append({
                "type": "web_search_20250305",
                "name": "web_search",
            })

        logger.info(
            "llm_call_start",
            model=self._model,
            web_search=self._enable_web_search,
            system_chars=len(system_prompt),
            user_chars=len(user_prompt),
        )

        try:
            response_text = self._call_with_retry(
                client=client,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tools=tools,
            )
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(
                f"Anthropic API call failed: {exc}",
                detail={"model": self._model},
            ) from exc

        logger.info(
            "llm_call_complete",
            model=self._model,
            response_chars=len(response_text),
        )
        return response_text

    # ── Private helpers ────────────────────────────────────────────────────────

    # @retry(
    #     retry=retry_if_exception_type(Exception),
    #     stop=stop_after_attempt(3),
    #     wait=wait_exponential(multiplier=1, min=2, max=8),
    #     reraise=True,
    # )
    def _call_with_retry(
        self,
        client: Any,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict],
    ) -> str:
        """
        Perform the actual Anthropic API call with tenacity retry decoration.

        @retry handles 429/529 transient errors automatically, retrying up to
        3 times with exponential backoff (2s, 4s, 8s).

        The web_search tool may cause the model to return multiple content blocks
        (text + tool_use + tool_result + text). We walk all blocks and
        concatenate the TEXT blocks to produce the final response.

        Args:
            client:        The instantiated Anthropic client.
            system_prompt: System role message.
            user_prompt:   User role message.
            tools:         List of tool dicts (may be empty).

        Returns:
            Concatenated text from all text-type content blocks.
        """
        # Build the API call kwargs
        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }

        if tools:
            call_kwargs["tools"] = tools

        response = client.messages.create(**call_kwargs)

        # Extract all text content blocks from the response
        # (web search responses have multiple blocks: tool_use + tool_result + text)
        text_blocks = [
            block.text
            for block in response.content
            if hasattr(block, "text") and block.type == "text"
        ]

        if not text_blocks:
            raise LLMError(
                "Anthropic API returned no text content blocks.",
                detail={"stop_reason": response.stop_reason},
            )

        # Join multiple text blocks (rare but possible with long web search chains)
        return "\n".join(text_blocks).strip()


    def _get_client(self) -> Any:
        """
        Lazily instantiate the Anthropic client.

        Lazy loading means tests that inject a mock don't need to have
        the anthropic package installed on the test runner if they
        replace the client before first call.
        """
        if self._client is None:
            # import anthropic

            if not self._api_key:
                raise LLMError(
                    "ANTHROPIC_API_KEY is not set. "
                    "Add it to your .env file to enable LLM features.",
                )

            # self._client = anthropic.Anthropic(api_key=self._api_key)
            logger.info("llm_client_initialised", model=self._model)

        return self._client

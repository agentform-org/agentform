"""LLM integration via LangChain."""

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agentform_runtime.logging_config import get_logger
from agentform_schema.ir import ResolvedAgent, ResolvedProvider


class LLMError(Exception):
    """Error during LLM execution."""

    pass


class LLMExecutor:
    """Executes LLM calls using LangChain."""

    def __init__(self, providers: dict[str, ResolvedProvider], verbose: bool = False):
        """Initialize LLM executor.

        Args:
            providers: Resolved provider configurations
            verbose: Enable verbose logging
        """
        self._providers = providers
        self._llm_cache: dict[str, Any] = {}
        self._verbose = verbose
        self._logger = get_logger("agentform_runtime.llm")

    def _get_llm(self, provider_name: str, model: str, params: dict[str, Any]) -> Any:
        """Get or create an LLM instance.

        Args:
            provider_name: Provider identifier
            model: Model name
            params: Model parameters

        Returns:
            LangChain LLM instance

        Raises:
            LLMError: If provider not found or not supported
        """
        cache_key = f"{provider_name}:{model}"
        if cache_key in self._llm_cache:
            self._logger.debug("llm_cache_hit", provider=provider_name, model=model)
            return self._llm_cache[cache_key]

        self._logger.debug("llm_creation_start", provider=provider_name, model=model, params=params)

        provider = self._providers.get(provider_name)
        if not provider:
            raise LLMError(f"Provider '{provider_name}' not found")

        api_key = provider.api_key.value
        if not api_key:
            raise LLMError(f"API key for provider '{provider_name}' not resolved")

        # Build params
        llm_params: dict[str, Any] = {
            "model": model,
            "api_key": api_key,
        }

        if params.get("temperature") is not None:
            llm_params["temperature"] = params["temperature"]
        if params.get("max_tokens") is not None:
            llm_params["max_tokens"] = params["max_tokens"]

        # Create LLM based on provider type
        # Handle both direct names (openai) and module-namespaced names (module.llm.openai)
        provider_type = self._extract_provider_type(provider_name)

        llm: ChatOpenAI | ChatAnthropic
        if provider_type == "openai":
            llm = ChatOpenAI(**llm_params)
        elif provider_type == "anthropic":
            llm = ChatAnthropic(**llm_params)
        else:
            raise LLMError(f"Unsupported provider: {provider_name}")

        self._llm_cache[cache_key] = llm
        self._logger.debug("llm_creation_complete", provider=provider_name, model=model)
        return llm

    def _extract_provider_type(self, provider_name: str) -> str:
        """Extract the provider type from a provider name.

        Handles both direct names and module-namespaced names:
        - "openai" -> "openai"
        - "module.llm.openai" -> "openai"
        - "module.my-module.anthropic" -> "anthropic"

        Args:
            provider_name: Provider name (may be namespaced)

        Returns:
            The base provider type (openai, anthropic, etc.)
        """
        # For module-namespaced providers, extract the last part
        # module.llm.openai -> openai
        # module.mymod.anthropic_custom -> anthropic_custom
        if provider_name.startswith("module."):
            parts = provider_name.split(".")
            if len(parts) >= 3:
                return parts[-1]
        return provider_name

    async def execute(
        self,
        agent: ResolvedAgent,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute an LLM call for an agent.

        Args:
            agent: Agent configuration
            input_data: Input data for the prompt

        Returns:
            Dict with 'response' and 'metadata' keys

        Raises:
            LLMError: If execution fails
        """
        self._logger.info(
            "llm_execution_start",
            agent_name=agent.name if hasattr(agent, "name") else "unknown",
            provider=agent.provider_name,
            model_preference=agent.model_preference,
            model_fallback=agent.model_fallback,
            has_instructions=bool(agent.instructions),
            input_keys=list(input_data.keys()) if input_data else [],
        )

        # Get params
        params = {}
        if agent.params:
            params = {
                "temperature": agent.params.temperature,
                "max_tokens": agent.params.max_tokens,
            }
            self._logger.debug("llm_params", params=params)

        # Try preferred model first, then fallback
        models_to_try = [agent.model_preference]
        if agent.model_fallback:
            models_to_try.append(agent.model_fallback)

        self._logger.debug("llm_models_to_try", models=models_to_try)

        last_error: Exception | None = None
        for model in models_to_try:
            try:
                self._logger.info("llm_model_attempt", model=model, provider=agent.provider_name)
                llm = self._get_llm(agent.provider_name, model, params)

                # Build messages
                messages: list[BaseMessage] = []
                if agent.instructions:
                    messages.append(SystemMessage(content=agent.instructions))
                    self._logger.debug(
                        "llm_system_message_added", instruction_length=len(agent.instructions)
                    )

                # Format input as user message
                if input_data:
                    import json

                    input_str = json.dumps(input_data, indent=2)
                    messages.append(HumanMessage(content=f"Input:\n{input_str}"))
                    self._logger.debug("llm_input_message_added", input_length=len(input_str))
                else:
                    messages.append(HumanMessage(content="Please proceed with your task."))
                    self._logger.debug("llm_empty_input_message_added")

                # Execute
                self._logger.debug("llm_invoke_start", model=model, message_count=len(messages))
                response = await llm.ainvoke(messages)
                self._logger.debug(
                    "llm_invoke_complete",
                    model=model,
                    response_length=len(response.content) if response.content else 0,
                )

                usage = getattr(response, "usage_metadata", None)
                result = {
                    "response": response.content,
                    "model": model,
                    "provider": agent.provider_name,
                    "usage": usage,
                }

                self._logger.info(
                    "llm_execution_success",
                    model=model,
                    provider=agent.provider_name,
                    response_length=len(response.content) if response.content else 0,
                    usage=usage,
                )

                return result

            except Exception as e:
                self._logger.warning(
                    "llm_model_failed",
                    model=model,
                    provider=agent.provider_name,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                last_error = e
                continue

        self._logger.error(
            "llm_execution_failed_all_models",
            models_tried=models_to_try,
            last_error=str(last_error) if last_error else None,
            last_error_type=type(last_error).__name__ if last_error else None,
        )
        raise LLMError(f"All models failed. Last error: {last_error}")

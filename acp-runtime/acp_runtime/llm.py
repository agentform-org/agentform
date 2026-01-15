"""LLM integration via LangChain."""

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from acp_schema.ir import ResolvedAgent, ResolvedProvider


class LLMError(Exception):
    """Error during LLM execution."""

    pass


class LLMExecutor:
    """Executes LLM calls using LangChain."""

    def __init__(self, providers: dict[str, ResolvedProvider]):
        """Initialize LLM executor.

        Args:
            providers: Resolved provider configurations
        """
        self._providers = providers
        self._llm_cache: dict[str, Any] = {}

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
            return self._llm_cache[cache_key]

        provider = self._providers.get(provider_name)
        if not provider:
            raise LLMError(f"Provider '{provider_name}' not found")

        api_key = provider.api_key.value
        if not api_key:
            raise LLMError(f"API key for provider '{provider_name}' not resolved")

        # Build params
        llm_params = {
            "model": model,
            "api_key": api_key,
        }

        if params.get("temperature") is not None:
            llm_params["temperature"] = params["temperature"]
        if params.get("max_tokens") is not None:
            llm_params["max_tokens"] = params["max_tokens"]

        # Create LLM based on provider
        if provider_name == "openai":
            llm = ChatOpenAI(**llm_params)
        elif provider_name == "anthropic":
            llm = ChatAnthropic(**llm_params)
        else:
            raise LLMError(f"Unsupported provider: {provider_name}")

        self._llm_cache[cache_key] = llm
        return llm

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
        # Get params
        params = {}
        if agent.params:
            params = {
                "temperature": agent.params.temperature,
                "max_tokens": agent.params.max_tokens,
            }

        # Try preferred model first, then fallback
        models_to_try = [agent.model_preference]
        if agent.model_fallback:
            models_to_try.append(agent.model_fallback)

        last_error: Exception | None = None
        for model in models_to_try:
            try:
                llm = self._get_llm(agent.provider_name, model, params)

                # Build messages
                messages = []
                if agent.instructions:
                    messages.append(SystemMessage(content=agent.instructions))

                # Format input as user message
                if input_data:
                    import json

                    input_str = json.dumps(input_data, indent=2)
                    messages.append(HumanMessage(content=f"Input:\n{input_str}"))
                else:
                    messages.append(HumanMessage(content="Please proceed with your task."))

                # Execute
                response = await llm.ainvoke(messages)

                return {
                    "response": response.content,
                    "model": model,
                    "provider": agent.provider_name,
                    "usage": getattr(response, "usage_metadata", None),
                }

            except Exception as e:
                last_error = e
                continue

        raise LLMError(f"All models failed. Last error: {last_error}")


from pydantic import BaseModel
from typing_extensions import override

from stark.core.commands_context import CommandsContext
from stark.core.commands_context_processor import CommandsContextLayer, CommandsContextProcessor
from stark.core.commands_manager import SearchResult
from stark.core.parsing import MatchResult, RecognizedEntity

# class LLMPrompt(str, Enum):
#     CHAT = "You're ..."


class Message(TypedDict):
    role: Literal["system", "user", "assistant", "developer", "tool"]
    content: str


# class LLMProvider(ABC):
#     @abstractmethod
#     async def complete(self, instructions: str, input: str) -> str: ...

#     @abstractmethod
#     async def complete_chat(self, messages: list[Message]) -> str: ...


# class OpenAIProvider(LLMProvider):
#     def __init__(self, api_key: str, default_model: str = "gpt-4o-mini", default_max_tokens: int = 256, base_url: str | None = None) -> None:
#         """
#         OpenAI-Compatible LLM Providers

#         Cloud / Hosted:
#         - Together AI       : Multi-model cloud service, OpenAI-compatible API.
#         - OpenRouter        : Proxy/router for multiple models, unified OpenAI-style endpoint.
#         - Mistral AI Cloud  : OpenAI-style REST API for proprietary or open-source models.

#         Local / Self-hosted:
#         - vLLM              : High-performance local serving; supports chat/completions/embeddings endpoints.
#         - Ollama            : Local LLM runner with OpenAI-compatible API.
#         - LiteLLM           : Abstraction layer for local or cloud models, OpenAI-compatible mode.

#         Notes:
#         - "OpenAI-compatible" means the server exposes the same HTTP endpoints, request/response JSON schemas as OpenAI.
#         - Enables reuse of OpenAI Python SDK or tooling without code changes.
#         - Some support only specific models or quantized weights; check docs for compatibility.
#         - Choose local/self-hosted for privacy/control; cloud for convenience and scale.
#         - To run a local model server from a gguf file, use `python3 -m llama_cpp.server --model <path_to_model.gguf>`
#         or `docker run --rm -it -p 8000:8000 -v /path/to/models:/models -e MODEL=/models/llama-model.gguf ghcr.io/abetlen/llama-cpp-python:latest`
#         """
#         from openai import AsyncOpenAI

#         self.client = AsyncOpenAI(api_key=api_key)
#         self.client.api_key = api_key
#         if base_url:
#             self.client.api_base = base_url
#         self.default_model = default_model
#         self.default_max_tokens = default_max_tokens

#     # LLMProviderProtocol

#     @override
#     async def complete(self, instructions: str, input: str) -> str:
#         resp = await self.client.responses.create(
#             model=self.default_model, instructions=instructions, input=input, max_tokens=self.default_max_tokens
#         )
#         return resp.choices[0].message["content"].strip()

#     @override
#     async def complete_chat(self, messages: list[Message]) -> str:
#         resp = await self.client.chat.completions.create(model=self.default_model, messages=messages, max_tokens=self.default_max_tokens)

# class LlamaProvider(LLMProvider):
#     def __init__(self, model_path: str, default_max_tokens: int = 256) -> None:
#         """
#         Supports any model compatible with llama.cpp, typically in GGUF format.
#         Includes LLaMA/LLaMA-2/LLaMA-3 families, their fine-tunes, and most
#         community GGUF models on Hugging Face. Also supports multimodal GGUF
#         models such as LLaVA and Moondream. Model must be a GGUF file and fit
#         your system’s memory.
#         """
#         from llama_cpp import Llama

#         self.model = asyncify(Llama(model_path=model_path))
#         self.default_max_tokens = default_max_tokens

#     # LLMProviderProtocol

#     @override
#     async def complete(self, instructions: str, input: str) -> str:
#         prompt = f"Precisely follow next instructions:\n{instructions}. And apply them to the input:\n{input}"
#         out = await asyncify(self.model)(prompt, max_tokens=self.default_max_tokens)  # TODO: stop flag
#         return out["choices"][0]["text"].strip()

#     @override
#     async def complete_chat(self, messages: list[Message]) -> str:
#         # TODO: implement:
#         # response_format={
#         #         "type": "json_object",
#         #         "schema": {
#         #             "type": "object",
#         #             "properties": {"team_name": {"type": "string"}},
#         #             "required": ["team_name"],
#         #         },
#         #     },
#         # TODO: also consider implementing Function Calling
#         return await asyncify(self.model.create_chat_completion)(messages)


class LLMCommandMatch(BaseModel):
    command_name: str
    substring: str
    # parameters: dict[str, str]


class LLMCommandMatches(BaseModel):
    matches: list[LLMCommandMatch]


class SimpleLLMCommandSearchProcessor(CommandsContextProcessor):
    # def __init__(self, llm_provider: LLMProvider):
    #     self.llm_provider = llm_provider

    # CommandsContextProcessor Implementation

    @override
    async def process_context_layer(
        self,
        string: str,
        context: CommandsContext,
        context_layer: CommandsContextLayer,
        recognized_entities: list[RecognizedEntity],
    ) -> list[SearchResult]: ...

    # Private

    async def _search_command(self, string: str, context_layer: CommandsContextLayer) -> list[SearchResult]:
        commands = json.dumps(context_layer.commands, cls=StarkJsonEncoder)
        prompt = f"Commands list is: {commands}"
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key="")
        response = await client.chat.completions.create(
            model="",
            messages=[Message(role="system", content=prompt), Message(role="user", content=string)],
            response_format={"type": "json", "model": LLMCommandMatches},
        )
        parsed_response: LLMCommandMatches = response.choices[0].message.content
        # response = await self.llm_provider.complete(instructions, f"Commands list is: {commands}\nUser input is '{string}'")
        # parse LLMMCommandMatch
        cmd_name_to_cmd = {cmd.name: cmd for cmd in context_layer.commands}
        return [
            SearchResult(
                cmd_name_to_cmd[match.command_name],
                MatchResult(
                    match.substring,
                    string.find(match.substring),
                    string.find(match.substring) + len(match.substring),
                    await self._recognize_parameters(match.substring),
                ),
            )
            for match in parsed_response.matches
        ]

    # async def _recognize_parameters(self, string: str, search_result: SearchResult) -> list[RecognizedEntity]: ...


# TODO: try split-take cmd name and params and single-take options
# TODO: try chat mode and it's support
# TODO: try structured output and it's support
# TODO: try function calling and it's support
# NOTE: llama cpp server supports those

# @CommandsManager().new("**", hidden=True)
# async def run_simple_llm_command(string: String, llm_provider: LLMProvider):
#     """Fallback command to ask/run (a big) LLM as a last resort when no other command is suitable for the given input."""
#     return Response.model_validate_strings(await llm_provider.complete("", string.value))

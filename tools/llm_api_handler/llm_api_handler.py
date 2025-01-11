"""llm_api_handler module for handling API calls to various LLM providers,
including reasoning chat completions for O1 with medium reasoning effort
and support for JSON mode, developer messages, tools, etc."""

import os
import base64
import json
from typing import List, Dict, Optional, Union, Any
import litellm
from litellm import completion


class llm_api_handler:
    """
    Main class for handling LLM API calls using litellm for various providers and tasks.
    
    Features:
      - Standard chat completion for any model (Anthropic, OpenAI, DeepSeek, etc.)
      - Reasoning-based chat completion for O1 model (default reasoning effort = medium),
        supporting JSON-mode, developer messages, function tools, etc.
      - PDF processing with base64 encoding (example usage with Anthropic).
      - Structured JSON completions with user-defined JSON schemas.
    """

    def __init__(self):
        """
        Initialize the LLM Orchestrator with API keys from config.
        This reads your environment-specific keys from config/config.py.
        """
        from config.config import (
            ANTHROPIC_API_KEY,
            PERPLEXITY_API_KEY,
            OPENAI_API_KEY,
            DEEPSEEK_API_KEY
        )

        os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
        os.environ["PERPLEXITY_API_KEY"] = PERPLEXITY_API_KEY
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY

        # Default settings
        self.default_max_tokens = 4096
        self.default_temperature = 0.7

        # Defaults for O1 reasoning
        self.o1_default_model = "o1"
        self.o1_default_reasoning_effort = "medium"

    def process_pdf(
        self,
        pdf_path: str,
        prompt: str,
        model: str = "anthropic/claude-3-5-sonnet-20241022",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Process a PDF file using the specified model.
        This includes encoding the PDF as base64 and sending it via `image_url` (PDF) to Anthropic (or similar model).

        Args:
            pdf_path (str): Path to the PDF file
            prompt (str): The prompt or instruction for processing the PDF
            model (str): Model to use (default: claude-3-5-sonnet-20241022)
            max_tokens (int, optional): Maximum tokens for the response
            temperature (float, optional): Temperature for response generation

        Returns:
            str: Model's free-form response after processing the PDF
        """
        # Encode PDF file to base64
        with open(pdf_path, "rb") as pdf_file:
            encoded_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")

        # Prepare messages with PDF content
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": f"data:application/pdf;base64,{encoded_pdf}"
                    }
                ]
            }
        ]

        # Make the LLM API call
        response = completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens or self.default_max_tokens,
            temperature=temperature or self.default_temperature
        )

        return response.choices[0].message.content

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str = "anthropic/claude-3-5-sonnet-20241022",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> Union[str, litellm.ModelResponse]:
        """
        Generic chat completion request to the specified model (e.g., Anthropic, OpenAI, etc.).

        Args:
            messages (List[Dict[str, Any]]): 
                A list of dicts with 'role' (e.g., "developer", "user") and 'content'.
                Content can be text or a list of text/image blocks for multimodal models.
            model (str): Which model to use (default: claude-3-5-sonnet-20241022)
            max_tokens (int, optional): Maximum tokens for the response
            temperature (float, optional): Temperature for response generation
            stream (bool): Whether to stream the response (returns a ModelResponse generator if True)

        Returns:
            str if stream=False, or a litellm.ModelResponse generator if stream=True
        """
        response = completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens or self.default_max_tokens,
            temperature=temperature or self.default_temperature,
            stream=stream
        )

        if stream:
            return response
        return response.choices[0].message.content

    def structured_json_completion(
        self,
        prompt: str,
        schema: Dict,
        model: str = "deepseek/deepseek-chat",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict:
        """
        Generate a structured (JSON) response using litellm's JSON-mode.

        Args:
            prompt (str): The user/content prompt to send to the LLM
            schema (Dict): A JSON schema dict describing the structure we expect
            model (str): The LLM model to use (default: deepseek/deepseek-chat)
            max_tokens (int, optional): Maximum tokens for the response
            temperature (float, optional): Temperature for response generation

        Returns:
            Dict: The LLM's JSON-parsed response.
                  If the LLM output is invalid JSON, you may need to handle exceptions or re-try logic.
        """
        messages = [{"role": "user", "content": prompt}]

        response = completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens or self.default_max_tokens,
            temperature=temperature or self.default_temperature,
            response_format={
                "type": "json_object",
                "response_schema": schema
            }
        )

        output_str = response.choices[0].message.content
        return json.loads(output_str)

    def reasoning_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        max_completion_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        response_format: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Union[str, Dict]] = None
    ) -> Union[str, litellm.ModelResponse]:
        """
        A specialized chat completion method for O1 models with default (medium) reasoning effort.
        Supports JSON mode, developer role messages, and function tools.

        Args:
            messages (List[Dict[str, Any]]): 
                A list of messages with roles (developer, user, etc.) and content 
                (which can be text or multiple text/image blocks).
            model (str, optional): O1 model ID to use. Defaults to self.o1_default_model.
            reasoning_effort (str, optional): 'low', 'medium', or 'high'. Defaults to self.o1_default_reasoning_effort.
            max_completion_tokens (int, optional): The upper bound of tokens for the completion.
            temperature (float, optional): Sampling temperature for the generation.
            stream (bool, optional): Whether to return a streaming response (generator).
            response_format (Dict, optional): 
                e.g. {"type": "json_object"} or {"type": "json_schema", "json_schema": {...}}.
            tools (List[Dict], optional): A list of function definitions for the model to optionally call.
            tool_choice (str or Dict, optional): Controls how/which tool is called 
                (e.g. 'none', 'auto', 'required', or {"type": "function", "function": {"name": "<your_function>"}}).

        Returns:
            Union[str, litellm.ModelResponse]: 
                A string if stream=False, or a ModelResponse generator if stream=True.
        """
        chosen_model = model if model else self.o1_default_model
        chosen_reasoning_effort = reasoning_effort if reasoning_effort else self.o1_default_reasoning_effort

        response = completion(
            model=chosen_model,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            stream=stream,
            reasoning_effort=chosen_reasoning_effort,  # O1-specific param
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice
        )

        if stream:
            # Return the streaming generator
            return response

        return response.choices[0].message.content


# -------------------------------------------------------------------------
# Example usage (not run by default):
# 
# if __name__ == "__main__":
#     handler = llm_api_handler()
#
#     messages_example = [
#         {
#             "role": "developer",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": """
#                     You are a helpful assistant that answers programming 
#                     questions in the style of a southern belle from the southeast United States.
#                     """
#                 }
#             ]
#         },
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "Are semicolons optional in JavaScript?"
#                 }
#             ]
#         }
#     ]
# 
#     # Example function (tool)
#     tools_example = [
#         {
#             "type": "function",
#             "function": {
#                 "name": "get_stock_price",
#                 "description": "Retrieves the current stock price for a given ticker symbol.",
#                 "parameters": {
#                     "type": "object",
#                     "required": ["ticker", "currency"],
#                     "properties": {
#                         "ticker": {
#                             "type": "string",
#                             "description": "The stock ticker symbol for the company."
#                         },
#                         "currency": {
#                             "type": "string",
#                             "description": "The currency in which to express the stock price (e.g. USD)."
#                         }
#                     },
#                     "additionalProperties": False
#                 },
#                 "strict": True
#             }
#         }
#     ]
#
#     # JSON schema example
#     math_schema = {
#         "name": "math_response",
#         "strict": True,
#         "schema": {
#             "type": "object",
#             "properties": {
#                 "operation": {
#                     "type": "string",
#                     "description": "The mathematical operation performed."
#                 },
#                 "inputs": {
#                     "type": "array",
#                     "description": "A list of numeric inputs.",
#                     "items": {"type": "number"}
#                 },
#                 "result": {
#                     "type": "number",
#                     "description": "The result of the operation."
#                 },
#                 "steps": {
#                     "type": "array",
#                     "description": "A sequence of steps to arrive at the result.",
#                     "items": {
#                         "type": "object",
#                         "properties": {
#                             "description": {"type": "string"},
#                             "value": {"type": "number"}
#                         },
#                         "required": ["description", "value"],
#                         "additionalProperties": False
#                     }
#                 }
#             },
#             "required": ["operation", "inputs", "result", "steps"],
#             "additionalProperties": False
#         }
#     ]
#
#     # Example usage:
#     response = handler.reasoning_chat_completion(
#         messages=messages_example,
#         response_format={"type": "json_schema", "json_schema": math_schema},
#         tools=tools_example,
#         tool_choice="auto",
#         max_completion_tokens=500
#     )
#
#     print("Response:", response)
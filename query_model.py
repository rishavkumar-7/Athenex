import google.generativeai as genai
from llama_index.llms.openai import OpenAI
from google.generativeai.types import FunctionDeclaration, Tool
from llama_index.core.llms import ChatMessage, MessageRole

from tools import tool_for_llm
from llama_index.core.base.llms.types import (
    ChatMessage, MessageRole,
)

def model_call(user_config: dict, query: str):
    """
    Calls the appropriate LLM based on user config with context-enriched query
    
    Args:
        user_config: Dictionary with provider, model_name, and api_key
        query: The query string (usually containing context from RAG)
        
    Returns:
        str: The generated response from the LLM
    """
    provider = user_config.get("provider")
    model_name = user_config.get("model_name")
    api_key = user_config.get("api_key")
    
    if not provider or not model_name:
        return "Error: Provider and model name must be configured"
    
    try:
        if provider.lower() == "gemini":
            if not api_key:
                return "Error: API key required for Gemini"

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)

            # Get tools for Gemini
            tool_dict = tool_for_llm()

            function_declarations = []
            for tool in tool_dict.values():
                descriptions = {
                    param: meta.get("description", "")
                    for param, meta in tool["parameters"]["properties"].items()
                }

                # Create a FunctionDeclaration from LlamaIndex-style tool
                fn_decl = FunctionDeclaration.from_function(
                    tool["function"],
                    descriptions=descriptions)
                function_declarations.append(fn_decl)

            tools = [Tool(function_declarations=function_declarations)]

            response = model.generate_content(
                query,
                tools=tools,
                tool_config={"function_calling_config": {"mode": "auto"}}
            )

            # Check if a function call was triggered
            if hasattr(response, 'candidates') and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call'):
                            function_name = part.function_call.name
                            function_args = part.function_call.args

                            for tool in tool_dict.values():
                                if tool["name"] == function_name:
                                    result = tool["fn"](**function_args)

                                    final_response = model.generate_content([
                                        {"text": query},
                                        {"text": f"Tool result: {result}"}
                                    ])
                                    return final_response.text

            return response.text

        elif provider.lower() == "openai":
            if not api_key:
                return "Error: API key required for OpenAI"
            
            llm = OpenAI(model=model_name, api_key=api_key)
            tools = tool_for_llm()
            response = llm.predict_and_call(
                tools,
                query,
                allow_parallel_tool_calls=True,
            )
            return response
            
        elif provider.lower() == "anthropic":
            return None
            
        elif provider.lower() == "local" or provider.lower() == "ollama":
            # for local set the llm only will use only one model for embedding
            usr_msg = ChatMessage(
                role=MessageRole.USER,
                content=query
                )
            # add model 
            response = model.chat_with_tools(
            user_msg=usr_msg,
            tools=tool_for_llm,
                tool_choice="choose summary tool for summarization",
                )
            return response.message.content
            
        else:
            return f"Error: Unsupported provider '{provider}'"
            
    except Exception as e:
        return f"Error calling {provider} model: {str(e)}"
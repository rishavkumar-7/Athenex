from typing import List, Literal, Optional
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.vector_stores import FilterCondition, MetadataFilters

# Import at function call time to avoid circular imports
def get_indices():
    from indexes import vector_index, summary_index
    return vector_index, summary_index

def tool_for_llm():
    """Returns a dictionary of all available tools, Gemini compatible"""
    tools = [vector_query_tool(), summary_query_tool()]
    tool_dict = {}

    for tool in tools:
        raw_schema = tool.metadata.fn_schema.model_json_schema()
        cleaned_schema = clean_schema(raw_schema)

        tool_dict[tool.metadata.name] = {
            "name": tool.metadata.name,
            "description": tool.metadata.description,
            "parameters": cleaned_schema,
            "function": tool.fn
        }
    print(tool_dict)
    return tool_dict


def vector_query_tool():
    """Create a vector query tool"""
    return FunctionTool.from_defaults(
        name="vector_query_tool",
        fn=vector_query,
        description="Search for specific information in documents using vector search"
    )

def summary_query_tool():
    """Create a summary query tool"""
    return FunctionTool.from_defaults(
        name="summary_query_tool",
        fn=summary_query,
        description="Get comprehensive summaries or overviews of document content"
    )

def vector_query(query: str, page_numbers: Optional[List[str]] = None) -> str:
    """Query the vector index for specific information"""
    vector_index, _ = get_indices()
    
    if vector_index is None:
        return "Error: Vector index not initialized. Please initialize the index first."

    try:
        if page_numbers:
            metadata_dicts = [{"key": "page_label", "value": p} for p in page_numbers]
            query_engine = vector_index.as_query_engine(
                similarity_top_k=2,
                filters=MetadataFilters.from_dicts(metadata_dicts, condition=FilterCondition.OR)
            )
        else:
            query_engine = vector_index.as_query_engine(similarity_top_k=2)

        response = query_engine.query(query)
        return str(response)
    except Exception as e:
        return f"Error querying vector index: {str(e)}"

def summary_query(query: str) -> str:
    """Generate a summary or overview based on document content"""
    _, summary_index = get_indices()
    
    if summary_index is None:
        return "Error: Summary index not initialized. Please initialize the index first."
    
    try:
        query_engine = summary_index.as_query_engine(
            response_mode="tree_summarize",
        )
        response = query_engine.query(query)
        return str(response)
    except Exception as e:
        return f"Error querying summary index: {str(e)}"
    
def clean_schema(schema: dict) -> dict:
    """Clean Pydantic schema for Gemini compatibility"""
    if isinstance(schema, dict):
        cleaned = {}
        for k, v in schema.items():
            if k in {"title", "description", "examples", "default"}:
                continue
            elif k == "anyOf":
                # Just return the first valid type Gemini might accept
                if isinstance(v, list) and v:
                    return clean_schema(v[0])
                continue
            cleaned[k] = clean_schema(v)
        return cleaned
    elif isinstance(schema, list):
        return [clean_schema(item) for item in schema]
    return schema

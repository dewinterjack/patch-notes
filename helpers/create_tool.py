from langchain_experimental.tools import PythonREPLTool


def create_tool(tool_type: str, **kwargs):
    """
    Create and configure a tool based on the specified type and parameters.

    Parameters:
    - tool_type (str): The type of tool to create (e.g., "python_repl").
    - kwargs: Additional parameters required for the tool's configuration.

    Returns:
    - tool: An initialized tool ready for use.
    """
    if tool_type == "python_repl":
        return PythonREPLTool()

    else:
        raise ValueError(f"Unsupported tool type: {tool_type}")

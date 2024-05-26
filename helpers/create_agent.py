from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


def create_agent(agent_name: str, tools: list, system_prompt: str):
    """
    Create and configure a LOL agent with the specified tools and system prompt.

    Parameters:
    - agent_name (str): The name of the agent.
    - tools (list): A list of tools the agent will use.
    - system_prompt (str): The system prompt to guide the agent's behavior.

    Returns:
    - AgentExecutor: An initialized agent ready for execution.
    """
    print(
        f"Creating agent: {agent_name} with tools: {tools} and system prompt: {system_prompt}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    print(f"Initialized LLM: {llm}")

    agent = create_openai_tools_agent(llm, tools, prompt)
    print(f"Created agent: {agent}")

    executor = AgentExecutor(agent=agent, tools=tools)
    print(f"Initialized AgentExecutor: {executor}")

    return executor

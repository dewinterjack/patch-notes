from dotenv import load_dotenv
load_dotenv()

# https://github.com/langchain-ai/langgraph/blob/main/examples/multi_agent/hierarchical_agent_teams.ipynb

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
import functools
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool


def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

members = ["Meta Specialist", "Match Analyst"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)

options = ["FINISH"] + members

function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

llm = ChatOpenAI(model="gpt-4o")

supervisor_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

@tool
def check_meta(query: str) -> str:
    """Check the meta."""
    return "Meta results"

@tool
def check_match(query: str) -> str:
    """Check the match."""
    return "Match results"


meta_agent = create_agent(llm, [check_meta], "You are a Meta Specialist.")
meta_node = functools.partial(agent_node, agent=meta_agent, name="Meta Specialist")

match_agent = create_agent(llm, [check_match], "You are a Match Analyst.")
match_node = functools.partial(agent_node, agent=match_agent, name="Match Analyst")

workflow = StateGraph(AgentState)
workflow.add_node("Meta Specialist", meta_node)
workflow.add_node("Match Analyst", match_node)
workflow.add_node("supervisor", supervisor_chain)

for member in members:
    workflow.add_edge(member, "supervisor")
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
workflow.set_entry_point("supervisor")

graph = workflow.compile()

for s in graph.stream(
    {
        "messages": [
            HumanMessage(content="What is strong in the meta right now?")
        ]
    },
    { "return_messages": 5 }
):
    if "__end__" not in s:
        print(s)
        print("----")
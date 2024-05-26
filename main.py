from dotenv import load_dotenv
from agents.meta import analyse_meta
from agents.coach import coach_chain
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from typing import Annotated, Sequence, TypedDict
import operator

load_dotenv()


# Define the agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage], operator.add]
    next: str


# Create the workflow
workflow = StateGraph(AgentState)
workflow.add_node("MetaAnalysisAgent", analyse_meta)
workflow.add_node("CoachAgent", coach_chain)

# Connect the nodes
members = ["MetaAnalysisAgent"]
for member in members:
    # Workers report back to the supervisor when done
    workflow.add_edge(member, "CoachAgent")

# The supervisor (CoachAgent) populates the "next" field in the graph state
# which routes to a node or finishes
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("CoachAgent", lambda x: x["next"], conditional_map)

# Add entry point
workflow.set_entry_point("CoachAgent")

# Compile the graph
graph = workflow.compile()


# Invoke the team
def run_workflow():
    initial_state = {
        "messages": [
            HumanMessage(content="What are the recent changes related to Skarner?")
        ]
    }
    for state in graph.stream(initial_state):
        if "__end__" not in state:
            print(state)
            print("----")


run_workflow()

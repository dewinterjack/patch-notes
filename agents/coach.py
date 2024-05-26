from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_openai import ChatOpenAI

system_prompt = (
    "You are a coach agent tasked with managing a conversation between the following workers: {members}. "
    "Given the following user request, respond with the worker to act next. Each worker will perform a task "
    "and respond with their results and status. When finished, respond with FINISH."
)

# Function for routing
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
                    {"enum": ["MetaAnalysisAgent", "FINISH"]},
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
            "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(["MetaAnalysisAgent", "FINISH"]), members="MetaAnalysisAgent")

llm = ChatOpenAI(model="gpt-4o")

# Create the supervisor chain
coach_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)

print("CoachAgent has been initialized and is ready to route tasks.")

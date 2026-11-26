from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from langchain_core.messages import SystemMessage

from llms import llm_open_ai as llm

class AgentState(TypedDict):
    query:str
    result:str


graph = StateGraph(AgentState)

EXAMPLES = [
    {
        "project": """
            We are looking for a Flutter developer to build a ride-sharing
            application with a FastAPI backend.
        """,
        "user_profile": """
            3 years Flutter experience.
            2 years FastAPI experience.
            Previously worked on a ride-booking application.
        """,
        "proposal": """
            Hi,

            I have strong experience with Flutter and FastAPI, and I've
            previously worked on a ride-booking application.

            I can help build the required mobile experience and integrate
            it with a scalable FastAPI backend.

            I'd be happy to discuss the project and understand the
            requirements in more detail.
        """
    }
]

system_prompt = SystemMessage(
    content=f"""
        You are a professional job proposal writer.

        Learn the expected proposal style from these examples:
        {EXAMPLES}
        
        Now write a job proposal for the client's project.

        Do not assume any skills or information that you don't have. Means don't add any content in the proposal that you don't know.
        
        
    """
)
 

# Nodes
def proposal_maker(state:AgentState):
    "This node is used for making proposal according to given user query."
    query = state['query']

    response = llm.invoke([
        system_prompt,
        ("human",f"query:{query}")
    ])

    return {"result":response.content}


# Add Nodes
graph.add_node("proposal_maker",proposal_maker)

# Add Edges
graph.add_edge(START,"proposal_maker")
graph.add_edge("proposal_maker",END)

# Build Graphs
app = graph.compile()


# Generating Graph Image
print("Generating graph image...")

# Save the Mermaid PNG to a file
png_data = app.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(png_data)
    print("Graph png has been generated successfully!")


from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from langchain_core.messages import SystemMessage

from llms import llm_open_ai as llm

class AgentState(TypedDict):
    job_description:str
    user_input:str
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
                
        You are a professional Upwork/Job-Application proposal writer.

        Your task is to write a short, natural, and effective proposal.

        STRICT FACTUAL RULES:

        1. Treat the user's profile as the ONLY source of truth about their experience.
        2. Never assume a skill based on the client's requirements.
        3. A technology mentioned in the job description does NOT mean the user has experience with it.
        4. Never invent years of experience, projects, clients, achievements, or technical expertise.
        5. If the user has no experience with a required technology, do not claim that they do.
        6. You may explain transferable experience, but clearly distinguish it from direct experience.
        7. If information is missing, simply omit it. Never fill gaps with assumptions.

        STYLE RULES:

        - Keep proposals short and direct.
        - Use simple, natural English.
        - Focus on the client's actual requirements.
        - Mention only relevant experience.
        - Avoid generic introductions and unnecessary details.

        EXAMPLES:
        Use the provided examples ONLY to learn writing style and structure.
        Never reuse their skills, technologies, projects, or experience as user facts.

        Before returning the proposal, check every factual statement against the user's profile.

    """
)
 

# Nodes
def proposal_maker(state:AgentState):
    "This node is used for making proposal according to given user query."
    user_input = state['user_input']
    job_description = state['job_description']

    response = llm.invoke([
        system_prompt,
        ("human", f"""
            User Input:
            {user_input}

            Job Description:
            {job_description}
        """)
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


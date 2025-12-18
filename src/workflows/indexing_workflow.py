"""
indexing Workflow - Sequential pipeline for repository indexing
"""
from google.adk.agents import SequentialAgent
from src.agents.index_agent import index_agent


indexing_workflow = SequentialAgent(
    name = "Indexing_workflow",
   sub_agents=[
        index_agent,
    ],
    description="Sequential pipeline for repository indexing"
)
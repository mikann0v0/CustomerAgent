import operator
from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    image_list: Annotated[List[str], operator.add]
    image_paths: Annotated[List[str], operator.add]
    references: Annotated[List[str], operator.add]
    top_k: int
    next_agent: str

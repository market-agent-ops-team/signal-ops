from typing import List, Dict, Any, TypedDict
from langgraph.graph import StateGraph,END,START
from agents.researcher_agent import news_research_node
from agents.editor_agent import editor_node

class MarketState(TypedDict):
    ticker:str
    target_date:str

    ml_trend:str
    ml_confidence:float

    news_items:List[Dict[str,str]]
    news_sentiments:str

    divergence_flag:bool
    analyst_reasoning:str

    final_report:str

def technical_research_node(state:MarketState):
    return{
        "ml_trend":"bullish",
        "ml_confidence":0.9
    }



def analyst_node(state:MarketState):
    return{
        "divergence_flag":False,
        "analyst_reasoning":""
    }




workflow = StateGraph(MarketState) # pyrefly: ignore[bad-specialization]

workflow.add_node("technical_research",technical_research_node)
workflow.add_node("news_research",news_research_node)
workflow.add_node("analyst",analyst_node)
workflow.add_node("editor",editor_node)

workflow.add_edge(START,"technical_research")
workflow.add_edge(START,"news_research")
    
workflow.add_edge("technical_research","analyst")
workflow.add_edge("news_research","analyst")

workflow.add_edge("analyst","editor")

workflow.add_edge("editor",END)

app = workflow.compile()

if __name__ == "__main__":
    # 1. Define the initial state (the input)
    initial_state = {
        "ticker": "RELIANCE.NS",
        "target_date": "2026-08-11"
    }

    # 2. Run the graph!
    print("Running Market-Agent-Ops Pipeline...\n")
    final_state = app.invoke(initial_state)

    # 3. Print the result
    print(final_state["final_report"])
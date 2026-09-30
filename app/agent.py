from typing import TypedDict
import re
import json

from langgraph.graph import StateGraph, END
from app.llm import chat
from app.rag import answer_faq
from app.tools import lookup_transaction
from app.tools import schedule_callback

class AgentState(TypedDict):
    message: str
    intent: str
    answer: str


INTENT_CLASSIFIER_PROMPT = (
    "Classify the caller's message into exactly one of these categories: "
    "faq, transaction_status, schedule_callback, other. "
    "Respond with exactly one word, nothing else.\n\n"
    "Examples:\n"
    "Message: What are your business hours? -> faq\n"
    "Message: What is the status of transaction TXN10005? -> transaction_status\n"
    "Message: Can I schedule a callback for tomorrow at 3pm? -> schedule_callback\n"
    "Message: Can you help me file my taxes? -> other\n"
)


def classify_intent(state):
    result = chat([
        {"role": "system", "content": INTENT_CLASSIFIER_PROMPT},
        {"role": "user", "content": state["message"]},
    ])
    intent = result.content.strip().lower()
    if intent not in {"faq", "transaction_status", "schedule_callback"}:
        intent = "other"
    return {"intent": intent}

def handle_faq(state):
    return {"answer": answer_faq(state["message"])}

def handle_transaction(state):
    match = re.search(r"TXN\d+", state["message"], re.IGNORECASE)
    if not match:
        return {"answer": "Could you tell me your transaction ID? It usually starts with TXN."}

    result = lookup_transaction(match.group())
    if result is None:
        return {"answer": "I couldn't find a transaction with that ID."}

    return {"answer": f"That transaction is {result['status']}, for ${result['amount']} on {result['date']}: {result['description']}."}

def handle_callback(state):
    extraction_prompt = (
        "Extract the caller's name, phone number, and preferred callback time "
        "from this message. Respond with ONLY a JSON object like "
        '{"name": "...", "phone_number": "...", "preferred_time": "..."}. '
        "Use null for any field you can't find.\n\n"
        f"Message: {state['message']}"
    )

    result = chat([{"role": "user", "content": extraction_prompt}])
    try:
        details = json.loads(result.content)
    except json.JSONDecodeError:
        details = {}

    if not details.get("name") or not details.get("phone_number") or not details.get("preferred_time"):
        return {"answer": "Could you give me your name, phone number, and a preferred time for the callback?"}

    confirmation = schedule_callback(details["name"], details["phone_number"], details["preferred_time"])
    return {"answer" : confirmation} 

def handle_fallback(state):
    return {
        "answer": "I can help with account questions, transaction status, or scheduling a callback. Could you rephrase your question?"
    }

def route_by_intent(state):
    return state["intent"]

graph = StateGraph(AgentState)
graph.add_node("classify", classify_intent)
graph.add_node("faq", handle_faq)
graph.add_node("transaction_status", handle_transaction)
graph.add_node("schedule_callback", handle_callback)
graph.add_node("other", handle_fallback)

graph.set_entry_point("classify")
graph.add_conditional_edges("classify", route_by_intent, {
    "faq": "faq",
    "transaction_status": "transaction_status",
    "schedule_callback": "schedule_callback",
    "other": "other",
})
graph.add_edge("faq", END)
graph.add_edge("transaction_status", END)
graph.add_edge("schedule_callback", END)
graph.add_edge("other", END)

compiled_agent = graph.compile()

def run_agent(message):
    result = compiled_agent.invoke({"message": message, "intent": "", "answer": ""})
    return result["answer"]
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parent
AI_DIR = ROOT / "src" / "ai"
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from procurement_assistant import ProcurementAssistant

EXAMPLES = [
    "Give me the purchasing summary",
    "Which orders need to be expedited?",
    "Which products are at risk of stockouts?",
    "Why are we ordering SKU-113?",
    "Analyze supplier SUP-001",
    "How much purchasing is planned for the next 30 days?",
]

@st.cache_resource(show_spinner=False)
def get_assistant():
    return ProcurementAssistant()


def show_procurement_ai():
    st.header("AI Procurement Assistant")
    st.caption("Ask questions about procurement planning data. This assistant uses rule-based analysis, not a generative AI model.")
    st.info("All purchasing amounts are proposals, not approved orders. Answers use the full planning dataset, independent of dashboard filters.")

    try:
        assistant = get_assistant()
    except Exception as exc:
        st.error("Could not initialize the procurement assistant.")
        st.exception(exc)
        return

    if "procurement_chat_history" not in st.session_state:
        st.session_state.procurement_chat_history = []

    if st.button("Clear conversation", key="clear_ai_conversation"):
        st.session_state.procurement_chat_history = []
        st.rerun()

    with st.expander("Example questions"):
        for i, question in enumerate(EXAMPLES):
            if st.button(question, key=f"ai_example_{i}", use_container_width=True):
                st.session_state.ai_pending_question = question

    for message in st.session_state.procurement_chat_history:
        with st.chat_message(message["role"]):
            st.text(message["content"])

    typed_question = st.chat_input("Ask a procurement question...")
    pending_question = st.session_state.pop("ai_pending_question", None)
    question = typed_question or pending_question
    if not question:
        return

    st.session_state.procurement_chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Analyzing planning data..."):
                answer = assistant.ask(question)
        except Exception as exc:
            answer = f"Unable to answer this question: {exc}"
        st.text(answer)

    st.session_state.procurement_chat_history.append({"role": "assistant", "content": answer})

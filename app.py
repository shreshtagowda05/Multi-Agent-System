import streamlit as st
from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain

st.set_page_config(page_title="Multi-Agent Research Assistant", page_icon="🔎", layout="wide")

# ---------- Sidebar ----------
with st.sidebar:
    st.title("🔎 Research Assistant")
    st.markdown(
        """
        **How it works**

        1. 🔍 **Search Agent** finds recent, reliable sources
        2. 📄 **Reader Agent** scrapes the best source
        3. ✍️ **Writer** drafts the report
        4. 🧐 **Critic** reviews the report
        """
    )
    st.divider()
    if st.button("Clear results", use_container_width=True):
        st.session_state.pop("state", None)
        st.session_state.pop("topic", None)
        st.rerun()


# ---------- Pipeline (same steps as pipeline.py, with live UI progress) ----------
def run_pipeline_with_ui(topic: str) -> dict:
    state = {}

    with st.status("Running research pipeline...", expanded=True) as status:
        # Step 1: search
        st.write("🔍 **Step 1** - Search agent is working...")
        search_agent = build_search_agent()
        search_result = search_agent.invoke({
            "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
        })
        state["search_results"] = search_result["messages"][-1].content
        st.write("✅ Search complete")

        # Step 2: reader
        st.write("📄 **Step 2** - Reader agent is scraping top resources...")
        reader_agent = build_reader_agent()
        reader_result = reader_agent.invoke({
            "messages": [("user",
                f"Based on the following search results about '{topic}', "
                f"pick the most relevant URL and scrape it for deeper content.\n\n"
                f"Search Results:\n{state['search_results'][:800]}"
            )]
        })
        state["scraped_content"] = reader_result["messages"][-1].content
        st.write("✅ Scraping complete")

        # Step 3: writer
        st.write("✍️ **Step 3** - Writer is drafting the report...")
        research_combined = (
            f"SEARCH RESULTS : \n {state['search_results']} \n\n"
            f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
        )
        state["report"] = writer_chain.invoke({
            "topic": topic,
            "research": research_combined,
        })
        st.write("✅ Report drafted")

        # Step 4: critic
        st.write("🧐 **Step 4** - Critic is reviewing the report...")
        state["feedback"] = critic_chain.invoke({"report": state["report"]})
        st.write("✅ Review complete")

        status.update(label="Pipeline complete!", state="complete", expanded=False)

    return state


def as_text(value) -> str:
    """Chain outputs may be str or a message object; normalise to text."""
    return value if isinstance(value, str) else getattr(value, "content", str(value))


# ---------- Main page ----------
st.title("Multi-Agent Research Assistant")
st.caption("Enter a topic and let the agents search, read, write and review for you.")

with st.form("research_form"):
    topic = st.text_input(
        "Research topic",
        placeholder="e.g. Latest advances in retrieval-augmented generation",
    )
    submitted = st.form_submit_button("Run research", type="primary")

if submitted:
    if not topic.strip():
        st.warning("Please enter a topic first.")
    else:
        try:
            st.session_state["state"] = run_pipeline_with_ui(topic.strip())
            st.session_state["topic"] = topic.strip()
        except Exception as e:
            st.error(f"Something went wrong while running the pipeline: {e}")

# ---------- Results ----------
state = st.session_state.get("state")
if state:
    st.divider()
    st.subheader(f"Results: {st.session_state.get('topic', '')}")

    report = as_text(state["report"])
    feedback = as_text(state["feedback"])

    tab_report, tab_feedback, tab_search, tab_scraped = st.tabs(
        ["📝 Final Report", "🧐 Critic Feedback", "🔍 Search Results", "📄 Scraped Content"]
    )

    with tab_report:
        st.markdown(report)
        st.download_button(
            "Download report (.md)",
            data=report,
            file_name="research_report.md",
            mime="text/markdown",
        )

    with tab_feedback:
        st.markdown(feedback)

    with tab_search:
        st.markdown(as_text(state["search_results"]))

    with tab_scraped:
        st.markdown(as_text(state["scraped_content"]))
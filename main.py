"""Agentic RAG with Web Search — Streamlit front end.

Upload a PDF, ask a question, and a three-agent CrewAI team answers it using
both your document (Qdrant + OpenAI embeddings) and the live web (Exa).

Credentials are supplied per-session in the sidebar, falling back to
``st.secrets`` / environment variables for local development. Nothing is
persisted server-side.
"""

from __future__ import annotations

import os
import sys

# --- Must run before anything imports crewai --------------------------------- #
# crewai pulls in chromadb, which refuses to load against sqlite3 < 3.35. Several
# hosts (Streamlit Community Cloud among them) still ship an older system sqlite,
# so swap in the modern bundled build when it is installed. Linux-only wheel, so
# a missing module here is normal on Windows/macOS.
try:  # pragma: no cover - environment dependent
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

# Keep agent runs from blocking on CrewAI's outbound telemetry in sandboxed
# hosts. Left narrow on purpose: AgentOps rides on OpenTelemetry, so the SDK as
# a whole must stay enabled for optional tracing to work.
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
# ----------------------------------------------------------------------------- #

import re
import uuid

import streamlit as st
from dotenv import load_dotenv

from crews import build_crew, run_crew
from qdrant_tool import load_pdf_into_qdrant

load_dotenv()

st.set_page_config(
    page_title="Agentic RAG with Web Search",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

AVAILABLE_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def secret(name: str) -> str:
    """Read a default credential from st.secrets, then the environment."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        # No secrets.toml present — perfectly normal.
        pass
    return os.getenv(name, "")


def slugify(value: str) -> str:
    """Turn a filename into a safe Qdrant collection name."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug[:40] or "document"


def init_state() -> None:
    defaults = {
        "messages": [],
        "pdf_loaded": False,
        "collection_name": "",
        "pdf_name": "",
        "chunk_count": 0,
        "processed_file_id": None,
        "session_id": uuid.uuid4().hex[:8],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def start_agentops(api_key: str) -> None:
    """Best-effort observability — never let it break the app."""
    if not api_key or st.session_state.get("agentops_started"):
        return
    try:
        import agentops

        agentops.init(
            api_key=api_key,
            default_tags=["agentic-rag", "crewai", "streamlit"],
            skip_auto_end_session=True,
            fail_safe=True,
        )
        st.session_state["agentops_started"] = True
    except Exception as exc:  # pragma: no cover - observability is optional
        st.sidebar.warning(f"AgentOps could not start: {exc}")


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar() -> dict[str, str]:
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.caption(
        "Keys are held in your browser session only — they are never stored "
        "server-side or logged."
    )

    with st.sidebar.expander("🔑 API Keys", expanded=True):
        openai_api_key = st.text_input(
            "OpenAI API Key",
            value=secret("OPENAI_API_KEY"),
            type="password",
            help="Used for embeddings (text-embedding-3-large) and the agent LLMs.",
        )
        qdrant_url = st.text_input(
            "Qdrant URL",
            value=secret("QDRANT_URL"),
            placeholder="https://xyz-example.aws.cloud.qdrant.io:6333",
            help="Your Qdrant Cloud cluster endpoint.",
        )
        qdrant_api_key = st.text_input(
            "Qdrant API Key",
            value=secret("QDRANT_API_KEY"),
            type="password",
        )
        exa_api_key = st.text_input(
            "Exa API Key",
            value=secret("EXA_API_KEY"),
            type="password",
            help="Powers the live web search agent.",
        )
        agentops_api_key = st.text_input(
            "AgentOps API Key (optional)",
            value=secret("AGENTOPS_API_KEY"),
            type="password",
            help="Optional tracing and monitoring of the agent run.",
        )

    model = st.sidebar.selectbox("🧠 Agent model", AVAILABLE_MODELS, index=0)

    st.sidebar.divider()
    st.sidebar.subheader("📄 Knowledge Base")

    uploaded_file = st.sidebar.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="The document becomes your private, searchable knowledge base.",
    )

    creds = {
        "openai_api_key": openai_api_key.strip(),
        "qdrant_url": qdrant_url.strip(),
        "qdrant_api_key": qdrant_api_key.strip(),
        "exa_api_key": exa_api_key.strip(),
        "agentops_api_key": agentops_api_key.strip(),
        "model": model,
    }

    handle_upload(uploaded_file, creds)

    if st.session_state.pdf_loaded:
        st.sidebar.success(
            f"**{st.session_state.pdf_name}**\n\n"
            f"{st.session_state.chunk_count} chunks indexed in "
            f"`{st.session_state.collection_name}`"
        )

    st.sidebar.divider()
    if st.sidebar.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    with st.sidebar.expander("ℹ️ Where do I get keys?"):
        st.markdown(
            "- [OpenAI](https://platform.openai.com/api-keys)\n"
            "- [Qdrant Cloud](https://cloud.qdrant.io/) — free 1 GB cluster\n"
            "- [Exa](https://dashboard.exa.ai/api-keys)\n"
            "- [AgentOps](https://app.agentops.ai/) — optional"
        )

    return creds


def handle_upload(uploaded_file, creds: dict[str, str]) -> None:
    """Ingest a newly uploaded PDF exactly once per file."""
    if uploaded_file is None:
        return

    file_id = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.processed_file_id == file_id:
        return

    missing = [
        label
        for label, value in (
            ("OpenAI API Key", creds["openai_api_key"]),
            ("Qdrant URL", creds["qdrant_url"]),
            ("Qdrant API Key", creds["qdrant_api_key"]),
        )
        if not value
    ]
    if missing:
        st.sidebar.error(
            "Add your " + ", ".join(missing) + " before uploading a PDF."
        )
        return

    collection_name = f"rag_{slugify(uploaded_file.name)}_{st.session_state.session_id}"

    with st.sidebar.status("Processing PDF...", expanded=True) as status:
        try:
            chunk_count = load_pdf_into_qdrant(
                pdf_file=uploaded_file,
                source_name=uploaded_file.name,
                collection_name=collection_name,
                openai_api_key=creds["openai_api_key"],
                qdrant_url=creds["qdrant_url"],
                qdrant_api_key=creds["qdrant_api_key"],
                recreate=True,
                progress_callback=lambda message: st.write(message),
            )
        except Exception as exc:
            status.update(label="PDF processing failed", state="error")
            st.error(str(exc))
            return

        st.session_state.update(
            pdf_loaded=True,
            collection_name=collection_name,
            pdf_name=uploaded_file.name,
            chunk_count=chunk_count,
            processed_file_id=file_id,
        )
        status.update(label="PDF indexed", state="complete", expanded=False)


# --------------------------------------------------------------------------- #
# Main panel
# --------------------------------------------------------------------------- #
def render_header() -> None:
    st.title("🤖 Agentic RAG with Web Search")
    st.markdown(
        "Ask a question and a **three-agent CrewAI team** answers it — searching "
        "your uploaded PDF *and* the live web, then synthesising both into one "
        "sourced response."
    )
    columns = st.columns(3)
    for column, (icon, title, body) in zip(
        columns,
        [
            ("🗄️", "DB Search Agent", "Semantic search over your PDF in Qdrant"),
            ("🌐", "Web Search Agent", "Real-time web results via Exa"),
            ("✍️", "Answer Agent", "Fuses both into a cited answer"),
        ],
    ):
        column.markdown(f"**{icon} {title}**  \n{body}")
    st.divider()


def missing_credentials(creds: dict[str, str]) -> list[str]:
    required = {
        "OpenAI API Key": creds["openai_api_key"],
        "Qdrant URL": creds["qdrant_url"],
        "Qdrant API Key": creds["qdrant_api_key"],
        "Exa API Key": creds["exa_api_key"],
    }
    return [label for label, value in required.items() if not value]


def answer_question(query: str, creds: dict[str, str]) -> str:
    crew = build_crew(
        openai_api_key=creds["openai_api_key"],
        exa_api_key=creds["exa_api_key"],
        qdrant_url=creds["qdrant_url"],
        qdrant_api_key=creds["qdrant_api_key"],
        collection_name=st.session_state.collection_name,
        model=creds["model"],
    )
    return run_crew(crew, query)


def main() -> None:
    init_state()
    creds = render_sidebar()
    start_agentops(creds["agentops_api_key"])
    render_header()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    blockers = missing_credentials(creds)
    if blockers:
        st.info(
            "👈 Add your **" + "**, **".join(blockers) + "** in the sidebar to begin."
        )
    elif not st.session_state.pdf_loaded:
        st.info("👈 Upload a PDF in the sidebar to build your knowledge base.")

    disabled = bool(blockers) or not st.session_state.pdf_loaded
    prompt = st.chat_input(
        "Ask a question about your document...", disabled=disabled
    )
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("The crew is working...", expanded=True) as status:
            st.write("🗄️ DB Search Agent is querying your PDF...")
            st.write("🌐 Web Search Agent is searching the web with Exa...")
            st.write("✍️ Answer Agent is composing the final answer...")
            try:
                answer = answer_question(prompt, creds)
                status.update(label="Answer ready", state="complete", expanded=False)
            except Exception as exc:
                status.update(label="The crew hit an error", state="error")
                answer = f"❌ Something went wrong while answering:\n\n```\n{exc}\n```"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()

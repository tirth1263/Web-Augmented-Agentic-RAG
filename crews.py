"""The CrewAI agent team behind the Agentic RAG app.

Three specialists run in a sequential process:

    DB Search Agent  ->  Web Search Agent  ->  Answer Agent

The DB agent mines the user's PDF (via Qdrant), the web agent gathers current
information (via Exa), and the answer agent fuses both into one response.
"""

from __future__ import annotations

from crewai import Agent, Crew, LLM, Process, Task
from crewai_tools import ExaSearchTool

from qdrant_tool import QdrantSearchTool

DEFAULT_MODEL = "gpt-4o-mini"


def build_llm(openai_api_key: str, model: str = DEFAULT_MODEL) -> LLM:
    """CrewAI routes through LiteLLM, so the model id is provider-prefixed."""
    return LLM(
        model=f"openai/{model}",
        api_key=openai_api_key,
        temperature=0.2,
    )


def build_agents(
    llm: LLM,
    qdrant_tool: QdrantSearchTool,
    exa_tool: ExaSearchTool,
) -> tuple[Agent, Agent, Agent]:
    db_search_agent = Agent(
        role="PDF Knowledge Base Researcher",
        goal=(
            "Find every passage in the user's uploaded PDF that bears on the "
            "question: {query}"
        ),
        backstory=(
            "You are a meticulous research librarian who works exclusively with "
            "the document the user provided. You search the vector database, "
            "quote what you find verbatim with page numbers, and you never "
            "invent material that is not in the document. If the PDF genuinely "
            "has nothing on the topic, you say so plainly rather than guessing."
        ),
        tools=[qdrant_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )

    web_search_agent = Agent(
        role="Web Research Specialist",
        goal=(
            "Gather accurate, current information from the open web about: {query}"
        ),
        backstory=(
            "You are an investigative researcher with a talent for finding the "
            "signal in a noisy web. You favour primary and authoritative sources, "
            "note when information is recent or contested, and always carry the "
            "source URL back with each claim so it can be verified."
        ),
        tools=[exa_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )

    answer_agent = Agent(
        role="Senior Research Analyst",
        goal=(
            "Write one clear, correct, well-sourced answer to: {query} — drawing "
            "on both the PDF findings and the web findings."
        ),
        backstory=(
            "You are the analyst who turns raw research into something a reader "
            "can actually use. You reconcile the document with the wider world, "
            "make it obvious which claim came from where, and flag disagreements "
            "between the two rather than papering over them. You write in clean "
            "Markdown and you never pad an answer to make it look thorough."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    return db_search_agent, web_search_agent, answer_agent


def build_tasks(
    db_search_agent: Agent,
    web_search_agent: Agent,
    answer_agent: Agent,
) -> list[Task]:
    db_task = Task(
        description=(
            "Search the uploaded PDF's Qdrant collection for material relevant to "
            "this question:\n\n{query}\n\n"
            "Use the Qdrant PDF Search tool. If the first search is thin, try one "
            "or two reworded queries before concluding the document is silent on "
            "the topic."
        ),
        expected_output=(
            "A bulleted digest of the relevant passages, each quoted or closely "
            "paraphrased and tagged with its page number. If the PDF contains "
            "nothing relevant, reply with exactly: NO_RELEVANT_PDF_CONTENT"
        ),
        agent=db_search_agent,
    )

    web_task = Task(
        description=(
            "Research this question on the web using the Exa search tool:\n\n{query}\n\n"
            "Prioritise authoritative and up-to-date sources. Capture concrete "
            "facts, figures and dates rather than vague summaries."
        ),
        expected_output=(
            "A bulleted digest of web findings. Each bullet states a fact and "
            "carries its source URL. Note publication dates where they matter."
        ),
        agent=web_search_agent,
    )

    answer_task = Task(
        description=(
            "Write the final answer to:\n\n{query}\n\n"
            "You have two research briefs in your context: one from the user's PDF "
            "and one from the web. Synthesise them into a single response.\n\n"
            "Rules:\n"
            "- Lead with a direct answer to the question, then expand.\n"
            "- Attribute every substantive claim: cite the PDF as (PDF, p. N) and "
            "the web with a Markdown link to the source.\n"
            "- Where the PDF and the web disagree, say so explicitly and give both.\n"
            "- If the PDF brief was NO_RELEVANT_PDF_CONTENT, answer from the web "
            "alone and state up front that the document did not cover this.\n"
            "- Do not invent citations. Do not restate the question back."
        ),
        expected_output=(
            "A polished Markdown answer: a direct opening answer, supporting detail "
            "with inline attribution, and a closing 'Sources' list separating "
            "document pages from web links."
        ),
        agent=answer_agent,
        context=[db_task, web_task],
        markdown=True,
    )

    return [db_task, web_task, answer_task]


def build_crew(
    openai_api_key: str,
    exa_api_key: str,
    qdrant_url: str,
    qdrant_api_key: str,
    collection_name: str,
    model: str = DEFAULT_MODEL,
    verbose: bool = True,
) -> Crew:
    """Assemble the full three-agent sequential crew."""
    llm = build_llm(openai_api_key, model=model)

    qdrant_tool = QdrantSearchTool(
        collection_name=collection_name,
        openai_api_key=openai_api_key,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
    )
    exa_tool = ExaSearchTool(api_key=exa_api_key, content=True, summary=True)

    db_search_agent, web_search_agent, answer_agent = build_agents(
        llm, qdrant_tool, exa_tool
    )
    tasks = build_tasks(db_search_agent, web_search_agent, answer_agent)

    return Crew(
        agents=[db_search_agent, web_search_agent, answer_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=verbose,
    )


def run_crew(crew: Crew, query: str) -> str:
    """Kick off the crew for one question and return the final answer text."""
    result = crew.kickoff(inputs={"query": query})
    return str(result)

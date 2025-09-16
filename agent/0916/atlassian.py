"""
LangGraph agent that ingests Confluence requirements and creates Jira issues.

The graph performs the following high-level stages:
1. Fetch and normalise the Confluence page (tables + images included).
2. Ask an LLM to analyse the requirements and highlight missing details.
3. Loop with the user for clarification until the requirements are clear.
4. Draft a Jira ticket and request approval before creating it via the API.

The agent relies on ``atlassian-python-api`` for Confluence/Jira access and any
LangChain-compatible chat (and optional vision) model for reasoning.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Optional, Sequence, Tuple, TypedDict

from atlassian import Confluence, Jira
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Interrupt


class AgentState(TypedDict, total=False):
    """Graph state that is threaded through the workflow."""

    messages: Annotated[List[BaseMessage], add_messages]
    confluence_payload: Dict[str, Any]
    requirement_brief: Dict[str, Any]
    outstanding_questions: List[str]
    clarification_history: List[str]
    ticket_draft: Dict[str, Any]
    ticket_key: Optional[str]
    params: Dict[str, Any]
    status: str


@dataclass
class ConfluenceToJiraConfig:
    """Static configuration for the agent run."""

    confluence_page_id: str
    jira_project_key: str
    issue_type: str = "Task"
    jira_field_overrides: Dict[str, Any] = field(default_factory=dict)
    additional_watchers: Sequence[str] = field(default_factory=tuple)
    dry_run: bool = False


@dataclass
class ConfluenceToJiraAgent:
    """LangGraph-based workflow that turns Confluence PRDs into Jira tickets."""

    confluence: Confluence
    jira: Jira
    llm: BaseLanguageModel
    vision_llm: Optional[BaseLanguageModel] = None

    def build(self) -> StateGraph[AgentState]:
        graph: StateGraph[AgentState] = StateGraph(AgentState)

        graph.add_node("fetch_confluence", self._fetch_confluence_node)
        graph.add_node("analyse_requirements", self._analyse_requirements_node)
        graph.add_node("clarify_with_user", self._clarify_with_user_node)
        graph.add_node("draft_ticket", self._draft_ticket_node)
        graph.add_node("confirm_ticket", self._confirm_ticket_node)
        graph.add_node("create_ticket", self._create_ticket_node)

        graph.add_edge(START, "fetch_confluence")
        graph.add_edge("fetch_confluence", "analyse_requirements")

        graph.add_conditional_edges(
            "analyse_requirements",
            self._needs_clarification,
            {
                "clarify": "clarify_with_user",
                "draft": "draft_ticket",
            },
        )

        graph.add_edge("clarify_with_user", "analyse_requirements")
        graph.add_edge("draft_ticket", "confirm_ticket")

        graph.add_conditional_edges(
            "confirm_ticket",
            self._approval_route,
            {
                "approved": "create_ticket",
                "revise": "clarify_with_user",
            },
        )

        graph.add_edge("create_ticket", END)

        return graph

    def initial_state(self, config: ConfluenceToJiraConfig) -> AgentState:
        """Construct the seed state that kicks off the workflow."""

        return {
            "params": {
                "confluence_page_id": config.confluence_page_id,
                "jira_project_key": config.jira_project_key,
                "issue_type": config.issue_type,
                "jira_field_overrides": config.jira_field_overrides,
                "additional_watchers": list(config.additional_watchers),
                "dry_run": config.dry_run,
            },
            "messages": [],
            "clarification_history": [],
            "status": "initialised",
        }

    def apply_clarification_response(
        self, state: AgentState, user_response: str, resolved: bool = True
    ) -> AgentState:
        """Merge a user reply to follow-up questions back into the state."""

        updated = dict(state)
        history = list(updated.get("clarification_history", []))
        history.append(user_response)
        updated["clarification_history"] = history
        if resolved:
            updated["outstanding_questions"] = []
        updated["status"] = "clarification_received"
        return updated

    def apply_approval_response(
        self, state: AgentState, approved: bool, feedback: Optional[str] = None
    ) -> AgentState:
        """Apply the approval decision collected outside the graph."""

        updated = dict(state)
        if approved:
            updated["status"] = "user_approved"
        else:
            updated["status"] = "user_rejected"
            if feedback:
                history = list(updated.get("clarification_history", []))
                history.append(f"Ticket rejected: {feedback}")
                updated["clarification_history"] = history
                updated["outstanding_questions"] = [feedback]
        return updated

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------

    def _fetch_confluence_node(self, state: AgentState) -> AgentState:
        params = state.get("params", {})
        page_id = params["confluence_page_id"]

        page = self.confluence.get_page_by_id(
            page_id,
            expand="body.storage,metadata.labels,version",
        )
        html = page["body"]["storage"]["value"]
        text, tables = _extract_text_and_tables(html)
        images = self._download_images(page_id)
        image_descriptions = self._describe_images(images)

        payload = {
            "title": page["title"],
            "version": page.get("version", {}).get("number"),
            "labels": [label["name"] for label in page.get("metadata", {}).get("labels", {}).get("results", [])],
            "raw_html": html,
            "plain_text": text,
            "tables": tables,
            "attachments": images,
            "image_descriptions": image_descriptions,
            "url": self.confluence.url + page.get("_links", {}).get("webui", ""),
        }

        updated = dict(state)
        updated["confluence_payload"] = payload
        updated.setdefault("clarification_history", [])
        updated.setdefault("messages", [])
        updated["status"] = "requirements_fetched"
        return updated

    def _analyse_requirements_node(self, state: AgentState) -> AgentState:
        payload = state["confluence_payload"]
        clarifications = state.get("clarification_history", [])

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert product analyst. Analyse the Confluence PRD, "
                    "understand requirements from text, tables, and images, and answer in JSON with keys:"
                    " summary, work_items (list), acceptance_criteria (list), open_questions (list)."
                    " Use Korean when the source is Korean."
                ),
                (
                    "human",
                    "Title: {title}\n"
                    "Labels: {labels}\n"
                    "Clarifications so far: {clarifications}\n"
                    "Plain text:\n{plain_text}\n\n"
                    "Tables:\n{tables}\n\n"
                    "Images:\n{image_descriptions}"
                ),
            ]
        )

        formatted_tables = json.dumps(payload["tables"], ensure_ascii=False, indent=2)
        formatted_images = json.dumps(payload["image_descriptions"], ensure_ascii=False, indent=2)
        prompt_value = prompt.format(
            title=payload["title"],
            labels=", ".join(payload["labels"]),
            clarifications="\n".join(clarifications) or "None",
            plain_text=payload["plain_text"],
            tables=formatted_tables,
            image_descriptions=formatted_images,
        )

        prompt_messages = prompt_value.to_messages()
        ai_reply = self.llm.invoke(prompt_messages)
        content = _coerce_to_text(ai_reply)

        try:
            structured = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "LLM response was not valid JSON. Received: {content}".format(content=content)
            ) from exc

        updated = dict(state)
        updated["requirement_brief"] = structured
        updated["outstanding_questions"] = structured.get("open_questions", [])
        updated.setdefault("messages", [])
        updated["messages"].extend([*prompt_messages, AIMessage(content=content)])
        updated["status"] = "requirements_analysed"
        return updated

    def _clarify_with_user_node(self, state: AgentState) -> AgentState:
        questions = state.get("outstanding_questions", [])
        if not questions:
            raise Interrupt({
                "type": "clarification",
                "message": "Clarification requested but no outstanding questions registered.",
                "questions": [],
            })

        raise Interrupt(
            {
                "type": "clarification",
                "questions": questions,
                "context": state.get("requirement_brief", {}),
            }
        )

    def _draft_ticket_node(self, state: AgentState) -> AgentState:
        params = state.get("params", {})
        structured = state["requirement_brief"]
        payload = state["confluence_payload"]

        summary = structured.get("summary", payload["title"])
        description_sections = ["*요약*\n" + structured.get("summary", "")]
        if structured.get("work_items"):
            description_sections.append("*작업 항목*\n" + "\n".join(f"- {item}" for item in structured["work_items"]))
        if structured.get("acceptance_criteria"):
            description_sections.append(
                "*수용 기준*\n" + "\n".join(f"# {crit}" for crit in structured["acceptance_criteria"])
            )
        description_sections.append(f"원본 문서: {payload['url']}")

        fields = {
            "project": {"key": params["jira_project_key"]},
            "issuetype": {"name": params.get("issue_type", "Task")},
            "summary": summary,
            "description": "\n\n".join(description_sections),
        }
        fields.update(params.get("jira_field_overrides", {}))

        updated = dict(state)
        updated["ticket_draft"] = fields
        updated["status"] = "ticket_drafted"
        return updated

    def _confirm_ticket_node(self, state: AgentState) -> AgentState:
        draft = state.get("ticket_draft")
        summary = draft.get("summary") if draft else None
        raise Interrupt(
            {
                "type": "approval",
                "ticket_preview": draft,
                "message": f"Create Jira issue '{summary}'?",
            }
        )

    def _create_ticket_node(self, state: AgentState) -> AgentState:
        params = state.get("params", {})
        draft = state["ticket_draft"]
        if params.get("dry_run"):
            updated = dict(state)
            updated["ticket_key"] = "DRY-RUN"
            updated["status"] = "dry_run_complete"
            return updated

        issue = self.jira.issue_create(fields=draft)
        ticket_key = issue.get("key") if isinstance(issue, dict) else getattr(issue, "key", None)
        issue_identifier = (
            issue.get("id") if isinstance(issue, dict) and issue.get("id") else ticket_key
        )

        watchers = params.get("additional_watchers", [])
        for watcher in watchers:
            if issue_identifier:
                self.jira.add_watcher(issue_identifier, watcher)

        updated = dict(state)
        updated["ticket_key"] = ticket_key
        updated["status"] = "ticket_created"
        return updated

    # ------------------------------------------------------------------
    # Conditional helpers
    # ------------------------------------------------------------------

    def _needs_clarification(self, state: AgentState) -> str:
        questions = [q for q in state.get("outstanding_questions", []) if q.strip()]
        if questions:
            return "clarify"
        return "draft"

    def _approval_route(self, state: AgentState) -> str:
        approval = state.get("status")
        if approval == "user_rejected":
            return "revise"
        return "approved"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _download_images(self, page_id: str) -> List[Dict[str, Any]]:
        attachments = self.confluence.get_attachments_from_content_id(page_id, limit=200)
        results = []
        for attachment in attachments.get("results", []):
            media_type = attachment.get("metadata", {}).get("mediaType", "")
            if not media_type.startswith("image/"):
                continue
            attachment_id = attachment["id"]
            binary = self.confluence.get_attachment_data(page_id, attachment_id)
            results.append(
                {
                    "title": attachment.get("title"),
                    "media_type": media_type,
                    "data": base64.b64encode(binary).decode("ascii"),
                }
            )
        return results

    def _describe_images(self, images: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
        if not images:
            return []
        if not self.vision_llm:
            return [
                {
                    "title": image["title"],
                    "description": "(No vision model configured. Provide manual context if critical.)",
                }
                for image in images
            ]

        descriptions = []
        for image in images:
            human = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "이 이미지를 설명해줘. 핵심 요구사항과 수치가 있다면 정리해." \
                                " Return Korean text if appropriate.",
                    },
                    {
                        "type": "image_base64",
                        "image_base64": image["data"],
                        "mime_type": image["media_type"],
                    },
                ]
            )
            reply = self.vision_llm.invoke([human])
            descriptions.append(
                {
                    "title": image["title"],
                    "description": _coerce_to_text(reply),
                }
            )
        return descriptions


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------


def _extract_text_and_tables(html: str) -> Tuple[str, List[Dict[str, Any]]]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError(
            "BeautifulSoup4 is required to parse Confluence tables. Install with 'pip install beautifulsoup4'."
        ) from exc

    soup = BeautifulSoup(html, "html.parser")
    tables_payload: List[Dict[str, Any]] = []
    for table_idx, table in enumerate(soup.find_all("table"), start=1):
        headers = [cell.get_text(strip=True) for cell in table.find_all("th")]
        rows = []
        for row in table.find_all("tr"):
            cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
            if headers and len(cells) == len(headers):
                rows.append(dict(zip(headers, cells)))
            elif cells:
                rows.append({f"col_{i}": value for i, value in enumerate(cells)})
        tables_payload.append({"index": table_idx, "headers": headers, "rows": rows})

    for tag in soup.find_all("table"):
        tag.decompose()

    text = soup.get_text("\n", strip=True)
    return text, tables_payload


def _coerce_to_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, AIMessage):
        return message.content if isinstance(message.content, str) else json.dumps(message.content)
    if isinstance(message, BaseMessage):
        return str(message.content)
    return str(message)


__all__ = [
    "AgentState",
    "ConfluenceToJiraConfig",
    "ConfluenceToJiraAgent",
]

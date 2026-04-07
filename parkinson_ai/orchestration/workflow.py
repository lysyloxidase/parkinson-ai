"""LangGraph workflow for Parkinson-AI multi-agent orchestration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

from parkinson_ai.agents.base import AgentResult
from parkinson_ai.agents.biomarker_interpreter import BiomarkerInterpreterAgent
from parkinson_ai.agents.drug_analyst import DrugAnalystAgent
from parkinson_ai.agents.genetic_counselor import GeneticCounselorAgent
from parkinson_ai.agents.imaging_analyst import ImagingAnalystAgent
from parkinson_ai.agents.kg_explorer import KGExplorerAgent
from parkinson_ai.agents.literature_agent import LiteratureAgent
from parkinson_ai.agents.risk_assessor import RiskAssessorAgent
from parkinson_ai.agents.router import RouterAgent
from parkinson_ai.agents.sentinel import SentinelAgent
from parkinson_ai.agents.staging_agent import StagingAgent
from parkinson_ai.data.clinicaltrials import ClinicalTrialsClient
from parkinson_ai.data.open_targets import OpenTargetsClient
from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph
from parkinson_ai.orchestration.decomposer import QueryDecomposer
from parkinson_ai.orchestration.state import AgentExecution, Task, WorkflowRuntimeState, WorkflowState

try:  # pragma: no cover - optional dependency path
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover - fallback when langgraph is unavailable
    END = "__end__"
    START = "__start__"
    StateGraph = None


class PDMultiAgentWorkflow:
    """Coordinator for the Parkinson-AI ten-agent workflow."""

    def __init__(
        self,
        *,
        graph: PDKnowledgeGraph | None = None,
        llm_client: Any | None = None,
        pubmed_client: Any | None = None,
        open_targets_client: OpenTargetsClient | None = None,
        clinical_trials_client: ClinicalTrialsClient | None = None,
    ) -> None:
        self.graph = graph or PDKnowledgeGraph()
        self.router = RouterAgent()
        self.decomposer = QueryDecomposer()
        self.sentinel = SentinelAgent(graph=self.graph)
        self.agents: dict[str, Any] = {
            "biomarker_interpreter": BiomarkerInterpreterAgent(graph=self.graph, llm_client=llm_client),
            "genetic_counselor": GeneticCounselorAgent(graph=self.graph, llm_client=llm_client),
            "imaging_analyst": ImagingAnalystAgent(graph=self.graph, llm_client=llm_client),
            "literature_agent": LiteratureAgent(graph=self.graph, pubmed_client=pubmed_client, llm_client=llm_client),
            "kg_explorer": KGExplorerAgent(graph=self.graph, llm_client=llm_client),
            "staging_agent": StagingAgent(llm_client=llm_client),
            "risk_assessor": RiskAssessorAgent(graph=self.graph, llm_client=llm_client),
            "drug_analyst": DrugAnalystAgent(
                graph=self.graph,
                llm_client=llm_client,
                open_targets_client=open_targets_client,
                clinical_trials_client=clinical_trials_client,
            ),
            "sentinel": self.sentinel,
        }
        self._compiled = self._build_graph()

    def invoke(self, query: str, *, patient_data: dict[str, Any] | None = None) -> WorkflowState:
        """Run the workflow and return the aggregated state."""

        initial_state: WorkflowRuntimeState = {
            "query": query,
            "patient_data": patient_data or {},
            "tasks": [],
            "results": [],
            "sentinel_report": {},
            "final_report": "",
        }
        if self._compiled is not None:
            final_state = self._compiled.invoke(initial_state)
        else:
            final_state = self._run_linear(initial_state)
        return WorkflowState(
            query=str(final_state["query"]),
            patient_data=dict(final_state.get("patient_data", {})),
            route=str(final_state.get("route", "")),
            tasks=list(final_state.get("tasks", [])),
            results=list(final_state.get("results", [])),
            sentinel_report=dict(final_state.get("sentinel_report", {})),
            final_report=str(final_state.get("final_report", "")),
        )

    def run(self, query: str, *, patient_data: dict[str, Any] | None = None) -> WorkflowState:
        """Alias for invoke for consistency with agent APIs."""

        return self.invoke(query, patient_data=patient_data)

    def _build_graph(self) -> Any | None:
        """Compile the LangGraph StateGraph when the dependency is available."""

        if StateGraph is None:
            return None
        graph_builder = StateGraph(WorkflowRuntimeState)
        graph_builder.add_node("route", self._route_node)
        graph_builder.add_node("specialists", self._specialists_node)
        graph_builder.add_node("sentinel", self._sentinel_node)
        graph_builder.add_node("aggregate", self._aggregate_node)
        graph_builder.add_edge(START, "route")
        graph_builder.add_edge("route", "specialists")
        graph_builder.add_edge("specialists", "sentinel")
        graph_builder.add_edge("sentinel", "aggregate")
        graph_builder.add_edge("aggregate", END)
        return graph_builder.compile()

    def _run_linear(self, state: WorkflowRuntimeState) -> WorkflowRuntimeState:
        """Run the graph nodes sequentially when LangGraph is unavailable."""

        routed = cast(WorkflowRuntimeState, {**state, **self._route_node(state)})
        specialists = cast(WorkflowRuntimeState, {**routed, **self._specialists_node(routed)})
        verified = cast(WorkflowRuntimeState, {**specialists, **self._sentinel_node(specialists)})
        return cast(WorkflowRuntimeState, {**verified, **self._aggregate_node(verified)})

    def _route_node(self, state: WorkflowRuntimeState) -> WorkflowRuntimeState:
        """Route the incoming query and decompose it into specialist tasks."""

        query = str(state["query"])
        patient_data = dict(state.get("patient_data", {}))
        routed = self.router.classify_task(query)
        tasks = self.decomposer.decompose(query, patient_data=patient_data)
        if routed.agent != "multi_agent" and (
            not tasks or all(task.agent == "router" for task in tasks)
        ):
            tasks = [Task(agent=routed.agent, description=query, payload={})]
        return {
            "route": routed.agent,
            "router_reason": routed.reason,
            "tasks": tasks,
        }

    def _specialists_node(self, state: WorkflowRuntimeState) -> WorkflowRuntimeState:
        """Execute specialist agents, using threads to parallelize independent tasks."""

        tasks = list(state.get("tasks", []))
        if not tasks:
            return {"results": []}
        max_workers = min(4, max(1, len(tasks)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._execute_task, task, dict(state.get("patient_data", {}))) for task in tasks]
            results = [future.result() for future in futures]
        return {"results": results}

    def _execute_task(self, task: Task, patient_data: dict[str, Any]) -> AgentExecution:
        """Execute a single decomposed task."""

        if task.agent == "router":
            routed = self.router.run(task.description)
            return AgentExecution(agent="router", content=routed.content, metadata=routed.metadata)
        agent = self.agents.get(task.agent)
        if agent is None:
            fallback = self.router.classify_task(task.description)
            agent = self.agents.get(fallback.agent)
            if agent is None:
                return AgentExecution(agent=task.agent, content="No matching agent.", metadata={"task": task.description})
        payload = dict(task.payload)
        if "patient_data" not in payload and patient_data:
            payload["patient_data"] = patient_data
        result: AgentResult = agent.run(task.description, **payload)
        return AgentExecution(agent=result.agent_name, content=result.content, metadata=result.metadata)

    def _sentinel_node(self, state: WorkflowRuntimeState) -> WorkflowRuntimeState:
        """Verify the specialist outputs."""

        agent_results = [AgentResult(agent_name=result.agent, content=result.content, metadata=result.metadata) for result in state.get("results", [])]
        combined_text = "\n".join(result.content for result in state.get("results", []))
        sentinel = self.sentinel.run(
            combined_text,
            agent_results=agent_results,
            patient_data=state.get("patient_data", {}),
        )
        return {"sentinel_report": sentinel.metadata}

    def _aggregate_node(self, state: WorkflowRuntimeState) -> WorkflowRuntimeState:
        """Aggregate specialist sections into a single final report."""

        sections = [f"Route: {state.get('route', '')}. {state.get('router_reason', '')}".strip()]
        for result in state.get("results", []):
            title = result.agent.replace("_", " ").title()
            sections.append(f"{title}: {result.content}")
        sentinel_report = dict(state.get("sentinel_report", {}))
        issues = sentinel_report.get("issues", [])
        if isinstance(issues, list) and issues:
            sections.append(f"Sentinel verification issues: {'; '.join(str(item) for item in issues)}")
        else:
            sections.append(f"Sentinel verification confidence: {sentinel_report.get('confidence_score', 'n/a')}.")
        return {"final_report": "\n\n".join(section for section in sections if section)}


def build_workflow(*args: Any, **kwargs: Any) -> PDMultiAgentWorkflow:
    """Build the Parkinson-AI multi-agent workflow."""

    return PDMultiAgentWorkflow(*args, **kwargs)

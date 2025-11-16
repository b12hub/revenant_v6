# agents/d_series/orchestrator.py
import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from core.agent_base import RevenantAgentBase
import logging
logging.basicConfig(level=logging.DEBUG)


class NodeStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class OrchestrationSchemaError(Exception):
    """Raised when workflow schema is invalid"""
    pass


class NodeExecutionError(Exception):
    """Raised when node execution fails"""
    pass


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass

class AgentCallFailedError(Exception):
    """Internal signal: agent execution failed but retry may occur."""
    pass


@dataclass
class NodeMetrics:
    node_id: str
    start_ts: float
    end_ts: Optional[float] = None
    duration_ms: Optional[float] = None
    status: NodeStatus = NodeStatus.PENDING
    error: Optional[str] = None
    retries: int = 0

    def to_dict(self):
        return {
            "node_id": self.node_id,
            "status": self.status.value if hasattr(self.status, "value") else self.status,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "duration_ms": self.duration_ms,
            "retries": self.retries,
            "error": self.error,
        }


@dataclass
class ExecutionContext:
    trace_id: str
    workflow_data: Dict[str, Any]
    node_results: Dict[str, Any]
    metrics: Dict[str, NodeMetrics]


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, reset_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures: Dict[str, int] = {}
        self.opened_until: Dict[str, float] = {}

    def can_execute(self, agent_id: str) -> bool:
        if agent_id in self.opened_until:
            if time.time() < self.opened_until[agent_id]:
                return False
            # Reset after timeout
            del self.opened_until[agent_id]
            self.failures[agent_id] = 0
        return True

    def record_failure(self, agent_id: str) -> bool:
        """
        Record a failure for agent_id. If the failure threshold is reached,
        mark circuit as opened for reset_timeout seconds.

        Returns:
            opened (bool): True if the circuit was opened by this call, False otherwise.
        """
        failures = self.failures.get(agent_id, 0) + 1
        self.failures[agent_id] = failures

        if failures >= self.failure_threshold:
            # open the circuit (do not raise here)
            self.opened_until[agent_id] = time.time() + self.reset_timeout
            return True

        return False

    def record_success(self, agent_id: str) -> None:
        """
        Reset the failure count for an agent when it succeeds.
        If the circuit was open, close it.
        """
        if agent_id in self.failures:
            self.failures[agent_id] = 0

        if agent_id in self.opened_until:
            del self.opened_until[agent_id]



class DSeriesOrchestrator(RevenantAgentBase):
    metadata = {
        "name": "DSeriesOrchestrator",
        "series": "d_series",
        "version": "0.1.0",
        "description": "Master orchestrator implementing the MetaCore workflow",
        "module": "agents.d_series.orchestrator",
        "workflow_path": "Revenant_MetaCore_v4_4_1_Multi-Series_Orchestration.json"
    }

    def __init__(
        self,
        registry: Optional[Any] = None,
        storage: Optional[Any] = None,
        executor: Optional[Callable] = None,
        logger: Optional[logging.Logger] = None,
    ):
        # ensure RevenantAgentBase gets required args
        super().__init__(name=self.metadata["name"], description=self.metadata["description"])

        self.registry = registry
        self.storage = storage or {}
        self.executor = executor
        self.logger = logger or logging.getLogger(__name__)

        self.workflow_spec: Optional[Dict[str, Any]] = None
        self.circuit_breaker = CircuitBreaker()
        self.max_concurrency = 10
        self.global_timeout = 300  # seconds
        self.semaphore = asyncio.Semaphore(self.max_concurrency)

        # attempt to load workflow at init (tests may patch Path.exists/open/json.load)
        self._load_workflow_spec()

    def _load_workflow_spec(self) -> None:
        """Load and validate workflow specification"""
        # absolute path (tests patch Path.exists/open)
        workflow_path = Path("/workflows/Revenant_MetaCore_v4_4_1_Multi-Series_Orchestration.json")

        if not workflow_path.exists():
            raise OrchestrationSchemaError(f"Workflow spec not found at {workflow_path}")

        try:
            with workflow_path.open("r") as fh:
                self.workflow_spec = json.load(fh)
            self._validate_workflow_schema()
        except json.JSONDecodeError as e:
            raise OrchestrationSchemaError(f"Invalid JSON in workflow spec: {e}")
        except Exception as e:
            raise OrchestrationSchemaError(f"Failed to load workflow spec: {e}")

    def _validate_workflow_schema(self) -> None:
        """Validate workflow schema structure"""
        if not isinstance(self.workflow_spec, dict):
            raise OrchestrationSchemaError("Workflow spec must be a JSON object")
        required_keys = {"name", "version", "nodes", "edges"}
        if not required_keys.issubset(set(self.workflow_spec.keys())):
            raise OrchestrationSchemaError("Workflow missing required keys: name, version, nodes, edges")

        for node in self.workflow_spec.get("nodes", []):
            if not isinstance(node, dict) or "id" not in node or "type" not in node:
                raise OrchestrationSchemaError("Node missing required fields: id or type")

    async def run(self, input_data):
        context = ExecutionContext(
            trace_id=input_data.get("trace_id"),
            workflow_data=input_data.get("workflow_data", {}),
            node_results={},
            metrics={}
        )

        try:
            result = await asyncio.wait_for(
                self._execute_workflow(context),
                timeout=self.global_timeout
            )

        except CircuitOpenError:
            raise

        except NodeExecutionError:
            raise

        except asyncio.TimeoutError:
            # ⭐ This must come BEFORE `Exception`
            raise NodeExecutionError("global timeout exceeded")

        except Exception as e:
            self.logger.error(
                "Workflow execution failed",
                extra={"trace_id": context.trace_id, "error": str(e)}
            )
            raise NodeExecutionError(str(e)) from e
        return { "node_results": context.node_results, "metrics": { nid: m.to_dict() for nid, m in context.metrics.items() } }


    async def _execute_workflow(self, context: ExecutionContext) -> Dict[str, Any]:
        """Execute the complete workflow — propagate node failures instead of swallowing them."""
        assert self.workflow_spec is not None, "workflow_spec must be loaded"

        nodes = self.workflow_spec.get("nodes", [])
        edges = self.workflow_spec.get("edges", [])

        # Build dependency graph
        dependencies = self._build_dependency_graph(edges)

        executed_nodes: Set[str] = set()
        node_map = {n["id"]: n for n in nodes}

        # Execute until all nodes have been processed or an error occurs
        while len(executed_nodes) < len(nodes):
            # Find ready nodes: not executed and all dependencies satisfied
            ready_nodes = [
                node_map[nid]
                for nid in node_map
                if nid not in executed_nodes and all(dep in executed_nodes for dep in dependencies.get(nid, []))
            ]

            if not ready_nodes:
                remaining = set(node_map.keys()) - executed_nodes
                raise NodeExecutionError(f"No ready nodes found, possible circular dependency. Remaining: {remaining}")

            # Launch ready nodes concurrently.
            # IMPORTANT: do NOT use return_exceptions=True — we want exceptions to bubble and stop execution.
            tasks = [self._execute_node(node, context) for node in ready_nodes]

            try:
                # If any task raises, gather will raise the first exception (default behavior).
                results = await asyncio.gather(*tasks)

            except Exception as e:

                # ------------------------------------
                # 1. Let CircuitOpenError bubble raw
                # ------------------------------------
                if isinstance(e, CircuitOpenError):
                    raise

                # ------------------------------------
                # 2. Let NodeExecutionError bubble raw
                # ------------------------------------
                if isinstance(e, NodeExecutionError):
                    raise

                # ------------------------------------
                # 3. Wrap unknown errors
                # ------------------------------------
                self.logger.error(
                    "Workflow execution halted due to unexpected error",
                    extra={"trace_id": context.trace_id, "error": str(e)}
                )
                raise NodeExecutionError(str(e)) from e

            # If we reach here, all tasks finished OK — mark them executed and store results
            for node, result in zip(ready_nodes, results):
                executed_nodes.add(node["id"])
                # node results are already stored inside _execute_node (context.node_results)
                # but ensure result presence just in case
                if node["id"] not in context.node_results:
                    context.node_results[node["id"]] = result

        # Format metrics for return
        metrics_out = {}
        for node_id, m in context.metrics.items():
            metrics_out[node_id] = {
                "status": m.status.value,
                "duration_ms": m.duration_ms,
                "retries": m.retries,
                "error": m.error,
            }

        return {
            "trace_id": context.trace_id,
            "node_results": context.node_results,
            "metrics": {
                nid: m.to_dict() if hasattr(m, "to_dict") else m.__dict__
                for nid, m in context.metrics.items()
            }
        }

    def _build_dependency_graph(self, edges: List[Dict[str, str]]) -> Dict[str, List[str]]:
        deps: Dict[str, List[str]] = {}
        for e in edges:
            src = e.get("source")
            tgt = e.get("target")
            if not src or not tgt:
                continue
            deps.setdefault(tgt, []).append(src)
        return deps

    async def _execute_node(self, node: Dict[str, Any], context: ExecutionContext) -> Any:
        node_id = node["id"]
        node_type = node["type"]
        config = node.get("config", {})

        metrics = NodeMetrics(node_id=node_id, start_ts=time.time())
        context.metrics[node_id] = metrics

        max_retries = int(config.get("max_retries", 0))
        timeout = float(config.get("timeout", 30))

        for attempt in range(max_retries + 1):
            try:
                metrics.status = NodeStatus.RUNNING
                metrics.retries = attempt

                async def _inner():
                    # REMOVE circuit check — handled inside _execute_call_agent
                    return await self._execute_node_by_type(node_type, config, context)

                result = await asyncio.wait_for(_inner(), timeout=timeout)

                metrics.status = NodeStatus.COMPLETED
                metrics.end_ts = time.time()
                metrics.duration_ms = (metrics.end_ts - metrics.start_ts) * 1000
                context.node_results[node_id] = result
                return result

            except CircuitOpenError as cb:
                metrics.error = str(cb)
                metrics.status = NodeStatus.FAILED

                # Propagate RAW CircuitOpenError — test 2 expects this
                raise

            except Exception as e:
                metrics.error = str(e)
                if attempt == max_retries:
                    metrics.status = NodeStatus.FAILED
                    raise NodeExecutionError(str(e)) from e

                await asyncio.sleep(2 ** attempt)

        metrics.status = NodeStatus.FAILED
        raise NodeExecutionError(f"Node {node_id} failed after {max_retries} retries")

    async def _execute_node_by_type(self, node_type: str, config: Dict[str, Any], context: ExecutionContext) -> Any:
        try:
            if node_type == "call_agent":
                return await self._execute_call_agent(config, context)
            if node_type == "evaluate":
                return await self._execute_evaluate(config, context)
            if node_type == "fuse":
                return await self._execute_fuse(config, context)
            if node_type == "store_memory":
                return await self._execute_store_memory(config, context)
            if node_type == "route":
                return await self._execute_route(config, context)
            if node_type == "external_http":
                return await self._execute_external_http(config, context)

            raise NodeExecutionError(f"Unknown node type: {node_type}")
        except Exception as e:
            # log and re-raise to let upper layer decide/retry
            self.logger.error(f"Node type '{node_type}' dispatch failed: {e}", extra={"trace_id": context.trace_id})
            raise

    async def _execute_call_agent(self, config: Dict[str, Any], context: ExecutionContext) -> Any:
        """Execute an agent call, enforcing circuit breaker behavior correctly."""

        agent_id = config.get("agent_id") or config.get("agent")
        if not agent_id:
            raise NodeExecutionError("call_agent missing 'agent_id'")

        # If circuit is open BEFORE execution → test expects CircuitOpenError
        if not self.circuit_breaker.can_execute(agent_id):
            raise CircuitOpenError(f"Circuit open for agent {agent_id}")

        async with self.semaphore:
            if not self.registry:
                raise NodeExecutionError("Agent registry not configured")

            agent_factory = self.registry.get_agent(agent_id)
            if not agent_factory:
                raise NodeExecutionError(f"Agent {agent_id} not found in registry")

            agent_instance = agent_factory()

            input_data = {
                **config.get("input", {}),
                "trace_id": context.trace_id,
                "workflow_data": context.workflow_data,
            }

            try:
                result = await agent_instance.run(input_data)
                # success -> reset failures
                self.circuit_breaker.record_success(agent_id)
                return result

            except Exception as e:
                opened = False
                try:
                    opened = self.circuit_breaker.record_failure(agent_id)
                except Exception:
                    self.logger.exception(
                        "Unexpected exception recording failure",
                        extra={"trace_id": context.trace_id, "agent_id": agent_id},
                    )

                if opened:
                    self.logger.warning(
                        f"Circuit for agent {agent_id} opened for {self.circuit_breaker.reset_timeout}s",
                        extra={"trace_id": context.trace_id, "agent_id": agent_id},
                    )

                # IMPORTANT: notify outer retry logic instead of returning/raising raw exception
                raise AgentCallFailedError(str(e))

    async def _execute_evaluate(self, config: Dict[str, Any], context: ExecutionContext) -> Any:
        if not self.registry:
            raise NodeExecutionError("Registry not available for evaluator")
        evaluator_factory = self.registry.get_agent("EvaluatorAgent")
        if not evaluator_factory:
            raise NodeExecutionError("EvaluatorAgent not found")
        evaluator_instance = evaluator_factory()
        payload = {
            "condition": config.get("condition"),
            "workflow_data": context.workflow_data,
            "node_results": context.node_results,
            "trace_id": context.trace_id,
        }
        return await evaluator_instance.run(payload)

    async def _execute_fuse(self, config: Dict[str, Any], context: ExecutionContext) -> Any:
        if not self.registry:
            raise NodeExecutionError("Registry not available for fusion")
        fusion_factory = self.registry.get_agent("FusionAgent")
        if not fusion_factory:
            raise NodeExecutionError("FusionAgent not found")
        fusion_instance = fusion_factory()
        payload = {
            "sources": config.get("sources", []),
            "strategy": config.get("strategy", "merge"),
            "workflow_data": context.workflow_data,
            "node_results": context.node_results,
            "trace_id": context.trace_id,
        }
        return await fusion_instance.run(payload)

    async def _execute_store_memory(self, config: Dict[str, Any], context: ExecutionContext) -> Any:
        if not self.registry:
            raise NodeExecutionError("Registry not available for memory operations")
        memory_factory = self.registry.get_agent("MemoryAgent")
        if not memory_factory:
            raise NodeExecutionError("MemoryAgent not found")
        memory_instance = memory_factory()
        payload = {
            "operation": "store",
            "key": config.get("key"),
            "value": context.workflow_data,
            "trace_id": context.trace_id,
        }
        return await memory_instance.run(payload)

    async def _execute_route(self, config: Dict[str, Any], context: ExecutionContext) -> Any:
        condition = config.get("condition")
        if condition and self._evaluate_condition(condition, context):
            return {"route": "condition_met", "next_nodes": config.get("true_branch", [])}
        return {"route": "condition_not_met", "next_nodes": config.get("false_branch", [])}

    async def _execute_external_http(self, config: Dict[str, Any], context: ExecutionContext) -> Any:
        url = config.get("url", "")
        method = config.get("method", "GET")
        self.logger.info(f"Mock HTTP call {method} {url}", extra={"trace_id": context.trace_id})
        await asyncio.sleep(0.05)
        return {"status": 200, "data": {"message": "mock"}}

    def _evaluate_condition(self, condition: str, context: ExecutionContext) -> bool:
        try:
            if "==" in condition:
                left, right = condition.split("==", 1)
                left = left.strip()
                right = right.strip().strip('"\'')

                value = context.workflow_data.get(left)
                return str(value) == right
            return False
        except Exception:
            return False

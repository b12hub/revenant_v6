# agents/d_series/coordinator_agent.py
"""
Coordinator Agent for Revenant Framework
D-Series: Orchestration and Fusion Layer
Handles task delegation, coordination, and result aggregation across agent networks.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any
from core.agent_base import RevenantAgentBase

logger = logging.getLogger(__name__)


class CoordinatorAgent(RevenantAgentBase):
    """Orchestrates multi-agent workflows and coordinates task execution."""


    metadata = {
        "name": "CoordinatorAgent",
        "version": "1.0.0",
        "series": "d_series",
        "description": "High-level task orchestration and agent coordination",
        "module": "agents.d_series.coordinator_agent"
    }
    def __init__(self, registry: Any):
        """
        Initialize CoordinatorAgent with agent registry.

        Args:
            registry: Agent registry service for discovering available agents
        """
        super().__init__(
            name=self.metadata['name'] ,
            description=self.metadata['description']
        )
        self.registry = registry
        self.active_tasks: Dict[str, List[str]] = {}
        self.task_counter = 0
        logger.info(f"Initialized {self.metadata['series']}-Series CoordinatorAgent v{self.metadata['version']}")

    async def collaborate(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main collaboration entry point for complex multi-agent tasks.

        Args:
            task: High-level task specification with type, requirements, and context

        Returns:
            Unified response with aggregated results and execution summary
        """
        logger.info(f"Starting collaboration for task type: {task.get('type', 'unknown')}")

        # Generate unique task ID
        task_id = f"task_{self.task_counter}_{task.get('type', 'general')}"
        self.task_counter += 1

        try:
            # Parse task and identify required agent types
            required_agents = self._analyze_task_requirements(task)
            logger.debug(f"Task requires agents: {required_agents}")

            # Discover available agents from registry
            available_agents = await self._discover_agents(required_agents)

            if not available_agents:
                return self._create_error_response(task_id, "No suitable agents available for task")

            # Route subtasks to appropriate agents
            subtask_results = await self.route_task(task, available_agents)

            # Aggregate and normalize results
            aggregated_results = self.aggregate_results(subtask_results)

            # Build final response
            response = {
                "task_id": task_id,
                "status": "completed",
                "results": aggregated_results,
                "summary": self._generate_execution_summary(task, available_agents, aggregated_results),
                "metadata": {
                    "agents_used": list(available_agents.keys()),
                    "subtasks_executed": len(subtask_results),
                    "coordinator_version": self.metadata['version']
                }
            }

            logger.info(f"Collaboration completed for task {task_id}")
            return response

        except Exception as e:
            logger.error(f"Collaboration failed for task {task_id}: {str(e)}")
            return self._create_error_response(task_id, f"Collaboration failed: {str(e)}")

    async def route_task(self, task: Dict[str, Any], agents: List[str]) -> List[Dict[str, Any]]:
        """
        Route subtasks to appropriate agents and execute in parallel where possible.

        Args:
            task: Original task specification
            agents: List of available agent identifiers

        Returns:
            List of results from all agent executions
        """
        subtasks = self._decompose_task(task, agents)
        logger.info(f"Routing {len(subtasks)} subtasks to agents")

        # Execute subtasks concurrently
        execution_tasks = []
        for subtask, agent_id in subtasks:
            execution_tasks.append(self._execute_subtask(subtask, agent_id))

        # Gather all results
        results = await asyncio.gather(*execution_tasks, return_exceptions=True)

        # Process results, handling exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Subtask {i} failed: {str(result)}")
                processed_results.append({
                    "agent": subtasks[i][1],
                    "status": "error",
                    "error": str(result),
                    "data": None
                })
            else:
                processed_results.append(result)

        return processed_results

    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate and normalize results from multiple agents.

        Args:
            results: List of agent response dictionaries

        Returns:
            Unified result structure with normalized data
        """
        if not results:
            return {"data": {}, "confidence": 0.0, "sources": []}

        successful_results = [r for r in results if r.get('status') == 'completed']

        # Merge data from all successful results
        aggregated_data = {}
        sources = []
        total_confidence = 0.0

        for result in successful_results:
            agent_data = result.get('data', {})
            aggregated_data.update(agent_data)

            source_info = {
                "agent": result.get('agent'),
                "timestamp": result.get('timestamp'),
                "confidence": result.get('confidence', 0.5)
            }
            sources.append(source_info)
            total_confidence += result.get('confidence', 0.5)

        # Calculate average confidence
        avg_confidence = total_confidence / len(successful_results) if successful_results else 0.0

        return {
            "data": aggregated_data,
            "confidence": avg_confidence,
            "sources": sources,
            "successful_agents": len(successful_results),
            "total_agents": len(results)
        }

    def _analyze_task_requirements(self, task: Dict[str, Any]) -> List[str]:
        """Analyze task to determine required agent types."""
        task_type = task.get('type', '').lower()
        requirements = task.get('requirements', [])

        agent_requirements = []

        # Map task types to agent series
        task_to_agent_map = {
            'iot': ['A', 'B'],
            'security': ['B'],
            'analysis': ['A', 'B'],
            'monitoring': ['A'],
            'response': ['B'],
            'complex': ['A', 'B']  # Requires both analysis and action
        }

        if task_type in task_to_agent_map:
            agent_requirements.extend(task_to_agent_map[task_type])

        # Add requirements from explicit specification
        if 'agents' in task:
            agent_requirements.extend(task['agents'])

        return list(set(agent_requirements))  # Remove duplicates

    async def _discover_agents(self, required_series: List[str]) -> Dict[str, Any]:
        """Discover available agents from registry."""
        available_agents = {}

        try:
            # Load registry (assuming JSON file)
            with open('registry.json', 'r') as f:
                registry_data = json.load(f)

            for agent_id, agent_info in registry_data.get('agents', {}).items():
                if agent_info.get('series') in required_series and agent_info.get('status') == 'active':
                    available_agents[agent_id] = agent_info

        except Exception as e:
            logger.warning(f"Registry discovery failed: {str(e)}")

        return available_agents

    def _decompose_task(self, task: Dict[str, Any], agents: Dict[str, Any]) -> List[tuple]:
        """Decompose main task into agent-specific subtasks."""
        subtasks = []
        task_type = task.get('type', 'general')

        for agent_id, agent_info in agents.items():
            agent_series = agent_info.get('series')
            subtask = self._create_subtask(task, agent_series, agent_id)
            subtasks.append((subtask, agent_id))

        return subtasks

    def _create_subtask(self, task: Dict[str, Any], agent_series: str, agent_id: str) -> Dict[str, Any]:
        """Create agent-specific subtask from main task."""
        base_subtask = {
            "parent_task_id": task.get('task_id'),
            "original_type": task.get('type'),
            "timestamp": task.get('timestamp'),
            "context": task.get('context', {})
        }

        # Customize based on agent series
        if agent_series == 'A':  # Analysis series
            base_subtask.update({
                "type": "analyze",
                "data": task.get('data'),
                "analysis_depth": task.get('analysis_depth', 'standard')
            })
        elif agent_series == 'B':  # Action series
            base_subtask.update({
                "type": "execute",
                "action_spec": task.get('action'),
                "parameters": task.get('parameters', {})
            })

        return base_subtask

    async def _execute_subtask(self, subtask: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
        """Execute a single subtask with specified agent."""
        # In real implementation, this would call the actual agent service
        # For now, simulate execution with mock response
        await asyncio.sleep(0.1)  # Simulate processing time

        return {
            "agent": agent_id,
            "status": "completed",
            "data": {"result": f"Mock execution by {agent_id}"},
            "confidence": 0.8,
            "timestamp": "2024-01-01T00:00:00Z"
        }

    def _generate_execution_summary(self, task: Dict[str, Any], agents: Dict[str, Any],
                                    results: Dict[str, Any]) -> str:
        """Generate human-readable execution summary."""
        return (f"Coordinated {len(agents)} agents for {task.get('type', 'unknown')} task. "
                f"Success rate: {results.get('successful_agents', 0)}/{results.get('total_agents', 0)} "
                f"with confidence: {results.get('confidence', 0.0):.2f}")

    def _create_error_response(self, task_id: str, error_msg: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "task_id": task_id,
            "status": "error",
            "results": {},
            "summary": error_msg,
            "metadata": {
                "error": error_msg,
                "coordinator_version": self.metadata['version']
            }
        }
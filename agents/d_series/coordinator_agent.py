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
        self.metadata = type(self).metadata.copy()

        super().__init__(
            name=self.metadata['name'],
            description=self.metadata['description']
        )
        self.registry = registry
        self.active_tasks: Dict[str, List[str]] = {}
        self.task_counter = 0

        logger.info(
            f"Initialized {CoordinatorAgent.metadata['series']}-Series FusionAgent v{CoordinatorAgent.metadata['version']}"
        )

    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate to collaborate for now."""
        return await self.collaborate(data)

    async def collaborate(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Main collaboration flow."""
        logger.info(f"Starting collaboration for task type: {task.get('type', 'unknown')}")

        task_id = f"task_{self.task_counter}_{task.get('type', 'general')}"
        self.task_counter += 1

        try:
            required_agents = self._analyze_task_requirements(task)
            logger.debug(f"Task requires agents: {required_agents}")

            available_agents = await self._discover_agents(required_agents)

            if not available_agents:
                return self._create_error_response(
                    task_id,
                    "No suitable agents available for task",
                    agents_used=[]
                )

            # Route & aggregate
            subtask_results = await self.route_task(task, available_agents)
            aggregated_results = self.aggregate_results(subtask_results)

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
            msg = f"Collaboration failed: {str(e)}"
            logger.error(msg)
            return self._create_error_response(task_id, msg, agents_used=[])

    async def route_task(self, task: Dict[str, Any], agents: Dict[str, Any]) -> List[Dict[str, Any]]:
        subtasks = self._decompose_task(task, agents)
        logger.info(f"Routing {len(subtasks)} subtasks to agents")

        execution_tasks = [
            self._execute_subtask(subtask, agent_id) for subtask, agent_id in subtasks
        ]

        results = await asyncio.gather(*execution_tasks, return_exceptions=True)

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
        if not results:
            return {"data": {}, "confidence": 0.0, "sources": []}

        successful_results = [r for r in results if r.get('status') == 'completed']

        aggregated_data = {}
        sources = []
        total_confidence = 0.0

        for result in successful_results:
            aggregated_data.update(result.get('data', {}))
            sources.append({
                "agent": result.get('agent'),
                "timestamp": result.get('timestamp'),
                "confidence": result.get('confidence', 0.5)
            })
            total_confidence += result.get('confidence', 0.5)

        avg_confidence = total_confidence / len(successful_results) if successful_results else 0.0

        return {
            "data": aggregated_data,
            "confidence": avg_confidence,
            "sources": sources,
            "successful_agents": len(successful_results),
            "total_agents": len(results)
        }

    def _analyze_task_requirements(self, task: Dict[str, Any]) -> List[str]:
        task_type = task.get('type', '').lower()
        requirements = task.get('requirements', [])

        agent_requirements = []

        task_to_agent_map = {
            'iot': ['A', 'B'],
            'security': ['B'],
            'analysis': ['A', 'B'],
            'monitoring': ['A'],
            'response': ['B'],
            'complex': ['A', 'B']
        }

        if task_type in task_to_agent_map:
            agent_requirements.extend(task_to_agent_map[task_type])

        if 'agents' in task:
            agent_requirements.extend(task['agents'])

        return list(set(agent_requirements))

    async def _discover_agents(self, required_series: List[str]) -> Dict[str, Any]:
        available_agents = {}

        try:
            with open('docs/registry.json', 'r') as f:
                registry_data = json.load(f)

            for agent_id, info in registry_data.get('agents', {}).items():
                if info.get('series') in required_series and info.get('status') == 'active':
                    available_agents[agent_id] = info

        except Exception as e:
            logger.warning(f"Registry discovery failed: {str(e)}")

        return available_agents

    def _decompose_task(self, task: Dict[str, Any], agents: Dict[str, Any]) -> List[tuple]:
        return [
            (self._create_subtask(task, info.get('series'), agent_id), agent_id)
            for agent_id, info in agents.items()
        ]

    def _create_subtask(self, task: Dict[str, Any], series: str, agent_id: str) -> Dict[str, Any]:
        base = {
            "parent_task_id": task.get('task_id'),
            "original_type": task.get('type'),
            "timestamp": task.get('timestamp'),
            "context": task.get('context', {})
        }

        if series == 'A':
            base.update({
                "type": "analyze",
                "data": task.get('data'),
                "analysis_depth": task.get('analysis_depth', 'standard')
            })
        elif series == 'B':
            base.update({
                "type": "execute",
                "action_spec": task.get('action'),
                "parameters": task.get('parameters', {})
            })

        return base

    async def _execute_subtask(self, subtask: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
        await asyncio.sleep(0.1)
        return {
            "agent": agent_id,
            "status": "completed",
            "data": {"result": f"Mock execution by {agent_id}"},
            "confidence": 0.8,
            "timestamp": "2024-01-01T00:00:00Z"
        }

    def _generate_execution_summary(self, task: Dict[str, Any], agents: Dict[str, Any], results: Dict[str, Any]) -> str:
        return (f"Coordinated {len(agents)} agents for {task.get('type', 'unknown')} task. "
                f"Success rate: {results.get('successful_agents', 0)}/{results.get('total_agents', 0)} "
                f"with confidence: {results.get('confidence', 0.0):.2f}")

    def _create_error_response(self, task_id: str, error_msg: str, agents_used: List[str]) -> Dict[str, Any]:
        return {
            "task_id": task_id,
            "status": "error",
            "results": {},
            "summary": error_msg,
            "metadata": {
                "error": error_msg,
                "coordinator_version": CoordinatorAgent.metadata['version'],
                "agents_used": agents_used
            }
        }

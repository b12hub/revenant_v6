# # /agents/a_series/optimization_agent.py
# from core.agent_base import RevenantAgentBase
# import asyncio
# from typing import Dict, List, Any
# import statistics
# from datetime import datetime, timedelta
#
#
# class OptimizationAgent(RevenantAgentBase):
#     def __init__(self):
#         super().__init__(
#             name="OptimizationAgent",
#             description="Continuously analyzes system performance and suggests optimizations for efficiency and cost reduction."
#         )
#         self.performance_data = {}
#         self.optimization_rules = {}
#
#     async def setup(self):
#         # Initialize performance tracking and optimization rules
#         self.performance_data = {
#             "agent_performance": {},
#             "resource_usage": {},
#             "cost_metrics": {}
#         }
#
#         self.optimization_rules = {
#             "response_time": {"threshold": 2.0, "action": "scale_up"},
#             "error_rate": {"threshold": 0.1, "action": "investigate"},
#             "cost_per_request": {"threshold": 0.05, "action": "optimize"}
#         }
#
#         await asyncio.sleep(0.1)
#
#     async def run(self, input_data: dict):
#         try:
#             # Collect performance metrics
#             metrics = await self._collect_metrics(input_data)
#
#             # Analyze for optimization opportunities
#             optimizations = await self._analyze_optimizations(metrics)
#
#             # Generate actionable recommendations
#             recommendations = await self._generate_recommendations(optimizations)
#
#             # Calculate potential savings
#             savings_analysis = await self._calculate_savings(recommendations)
#
#             result = {
#                 "optimization_opportunities": len(optimizations),
#                 "recommendations": recommendations,
#                 "estimated_savings": savings_analysis,
#                 "performance_metrics": metrics,
#                 "optimization_score": self._calculate_optimization_score(optimizations),
#                 "analysis_timestamp": datetime.now().isoformat()
#             }
#
#             return {
#                 "agent": self.name,
#                 "status": "ok",
#                 "summary": f"Optimization analysis complete - {len(optimizations)} opportunities found",
#                 "data": result
#             }
#
#         except Exception as e:
#             return await self.on_error(e)
#
#     async def _collect_metrics(self, input_data: dict) -> Dict[str, Any]:
#         agent_type = input_data.get("agent_type", "unknown")
#         execution_time = input_data.get("duration_ms", 0)
#         quality_score = input_data.get("quality_score", 0)
#         success = input_data.get("status") in ["completed", "success"]
#
#         # Update agent-specific performance data
#         if agent_type not in self.performance_data["agent_performance"]:
#             self.performance_data["agent_performance"][agent_type] = {
#                 "response_times": [],
#                 "success_rates": [],
#                 "quality_scores": []
#             }
#
#         agent_data = self.performance_data["agent_performance"][agent_type]
#         agent_data["response_times"].append(execution_time)
#         agent_data["success_rates"].append(1.0 if success else 0.0)
#         agent_data["quality_scores"].append(quality_score)
#
#         # Keep only recent data (last 100 points)
#         for key in ["response_times", "success_rates", "quality_scores"]:
#             if len(agent_data[key]) > 100:
#                 agent_data[key] = agent_data[key][-100:]
#
#         return {
#             "agent_type": agent_type,
#             "current_response_time": execution_time,
#             "current_quality": quality_score,
#             "success": success,
#             "historical_data": agent_data
#         }
#
#     async def _analyze_optimizations(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
#         optimizations = []
#         agent_type = metrics["agent_type"]
#         historical_data = metrics["historical_data"]
#
#         # Analyze response time optimization
#         if historical_data["response_times"]:
#             avg_response = statistics.mean(historical_data["response_times"])
#             if avg_response > self.optimization_rules["response_time"]["threshold"] * 1000:  # Convert to ms
#                 optimizations.append({
#                     "type": "performance",
#                     "metric": "response_time",
#                     "current_value": avg_response,
#                     "threshold": self.optimization_rules["response_time"]["threshold"] * 1000,
#                     "suggestion": f"Optimize {agent_type} response time - currently {avg_response:.2f}ms"
#                 })
#
#         # Analyze error rate optimization
#         if historical_data["success_rates"]:
#             error_rate = 1.0 - statistics.mean(historical_data["success_rates"])
#             if error_rate > self.optimization_rules["error_rate"]["threshold"]:
#                 optimizations.append({
#                     "type": "reliability",
#                     "metric": "error_rate",
#                     "current_value": error_rate,
#                     "threshold": self.optimization_rules["error_rate"]["threshold"],
#                     "suggestion": f"Reduce {agent_type} error rate - currently {error_rate:.2%}"
#                 })
#
#         # Analyze quality optimization
#         if historical_data["quality_scores"]:
#             avg_quality = statistics.mean(historical_data["quality_scores"])
#             if avg_quality < 7.0:  # Quality threshold
#                 optimizations.append({
#                     "type": "quality",
#                     "metric": "quality_score",
#                     "current_value": avg_quality,
#                     "threshold": 7.0,
#                     "suggestion": f"Improve {agent_type} output quality - currently {avg_quality:.1f}/10"
#                 })
#
#         return optimizations
#
#     async def _generate_recommendations(self, optimizations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """
#         :param optimizations:
#         :return:
#         """
#         recommendations = []
#
#         for opt in optimizations:
#             if opt["type"] == "performance":
#                 recommendations.append({
#                     "priority": "high" if opt["current_value"] > opt["threshold"] * 2 else "medium",
#                     "action": "Optimize algorithm or cache frequently used data",
#                     "impact": "reduced_latency",
#                     "effort": "medium"
#                 })
#             elif opt["type"] == "reliability":
#                 recommendations.append({
#                     "priority": "high",
#                     "action": "Add error handling and retry mechanisms",
#                     "impact": "improved_reliability",
#                     "effort": "low"
#                 })
#             elif opt["type"] == "quality":
#                 recommendations.append({
#                     "priority": "medium",
#                     "action": "Enhance validation and quality checks",
#                     "impact": "better_outputs",
#                     "effort": "high"
#                 })
#
#         # Add general recommendations if no specific optimizations found
#         if not recommendations:
#             recommendations.append({
#                 "priority": "low",
#                 "action": "Continue monitoring system performance",
#                 "impact": "maintenance",
#                 "effort": "low"
#             })
#
#         return recommendations
#
#     async def _calculate_savings(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
#         cost_savings = 0
#         time_savings = 0
#
#         for rec in recommendations:
#             if rec["impact"] == "reduced_latency":
#                 time_savings += 0.5  # Estimated 0.5 seconds per request
#                 cost_savings += 0.01  # Estimated cost savings per request
#             elif rec["impact"] == "improved_reliability":
#                 cost_savings += 0.02  # Reduced error handling costs
#             elif rec["impact"] == "better_outputs":
#                 cost_savings += 0.005  # Reduced rework costs
#
#         return {
#             "estimated_cost_savings_per_request": cost_savings,
#             "estimated_time_savings_per_request": time_savings,
#             "monthly_savings_estimate": cost_savings * 10000,  # Assume 10k requests/month
#             "confidence": "medium"
#         }
#
#     def _calculate_optimization_score(self, optimizations: List[Dict[str, Any]]) -> float:
#         if not optimizations:
#             return 10.0  # Perfect score if no optimizations needed
#
#         base_score = 10.0
#         penalty = len(optimizations) * 1.5  # Penalty for each optimization needed
#
#         # Additional penalty for high-priority issues
#         high_priority_count = sum(1 for opt in optimizations if opt.get("priority") == "high")
#         penalty += high_priority_count * 2.0
#
#         return max(0.0, base_score - penalty)


# /agents/a_series/optimization_agent.py
from core.agent_base import RevenantAgentBase
import asyncio
from typing import Dict,  Any
from datetime import datetime

class OptimizationAgent(RevenantAgentBase):
    """
    Continuously analyzes system performance and suggests optimizations for efficiency and cost reduction.

    Input:
        - agent_type (str): Agent being analyzed
        - duration_ms (int): Execution duration
        - quality_score (float): Quality score
        - status (str): Execution status

    Output:
        - optimization_opportunities (int): Number of optimization opportunities found
        - recommendations (list): Actionable optimization recommendations
        - estimated_savings (dict): Estimated cost/time savings
        - optimization_score (float): Overall optimization score (0-10)
    """

    metadata = {
        "name": "OptimizationAgent",
        "series": "a_series",
        "version": "0.1.0",
        "description": "Continuously analyzes system performance and suggests optimizations for efficiency and cost reduction."
    }

    def __init__(self):
        super().__init__(
            name="OptimizationAgent",
            description="Continuously analyzes system performance and suggests optimizations for efficiency and cost reduction."
        )
        self.performance_data = {}
        self.optimization_rules = {}
        self.metadata = OptimizationAgent.metadata

    async def setup(self):
        # Initialize performance tracking and optimization rules
        self.performance_data = {
            "agent_performance": {},
            "resource_usage": {},
            "cost_metrics": {}
        }

        self.optimization_rules = {
            "response_time": {"threshold": 2.0, "action": "scale_up"},
            "error_rate": {"threshold": 0.1, "action": "investigate"},
            "cost_per_request": {"threshold": 0.05, "action": "optimize"}
        }

        await asyncio.sleep(0.1)

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Collect performance metrics
            metrics = await self._collect_metrics(input_data)

            # Analyze for optimization opportunities
            optimizations = await self._analyze_optimizations(metrics)

            # Generate actionable recommendations
            recommendations = await self._generate_recommendations(optimizations)

            # Calculate potential savings
            savings_analysis = await self._calculate_savings(recommendations)

            result = {
                "optimization_opportunities": len(optimizations),
                "recommendations": recommendations,
                "estimated_savings": savings_analysis,
                "performance_metrics": metrics,
                "optimization_score": self._calculate_optimization_score(optimizations),
                "analysis_timestamp": datetime.now().isoformat()
            }

            return {
                "agent": self.name,
                "status": "ok",
                "summary": f"Optimization analysis complete - {len(optimizations)} opportunities found",
                "data": result
            }

        except Exception as e:
            return await self.on_error(e)

# ... existing code ...
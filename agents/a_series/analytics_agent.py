# # /agents/a_series/analytics_agent.py
# from core.agent_base import RevenantAgentBase
# import asyncio
# from typing import Dict, List, Any
# from datetime import datetime, timedelta
# import statistics
#
#
# class AnalyticsAgent(RevenantAgentBase):
#     def __init__(self):
#         super().__init__(
#             name="AnalyticsAgent",
#             description="Analyzes system performance, agent effectiveness, and provides data-driven insights for optimization."
#         )
#         self.metrics_history = {}
#         self.performance_thresholds = {
#             "response_time": 5.0,  # seconds
#             "success_rate": 0.95,  # 95%
#             "quality_score": 7.0  # out of 10
#         }
#
#     async def setup(self):
#         # Initialize analytics storage and thresholds
#         self.metrics_history = {
#             "response_times": [],
#             "success_rates": [],
#             "quality_scores": [],
#             "agent_performance": {}
#         }
#         await asyncio.sleep(0.1)
#
#     async def run(self, input_data: dict):
#         try:
#             # Extract metrics from input
#             metrics = self._extract_metrics(input_data)
#
#             # Analyze performance trends
#             trend_analysis = await self._analyze_trends(metrics)
#
#             # Generate insights and recommendations
#             insights = await self._generate_insights(trend_analysis)
#
#             # Calculate system health score
#             health_score = self._calculate_health_score(trend_analysis)
#
#             result = {
#                 "system_health_score": health_score,
#                 "performance_trends": trend_analysis,
#                 "key_insights": insights["key_insights"],
#                 "recommendations": insights["recommendations"],
#                 "anomalies_detected": trend_analysis["anomalies"],
#                 "timestamp": datetime.now().isoformat()
#             }
#
#             return {
#                 "agent": self.name,
#                 "status": "ok",
#                 "summary": f"Analytics completed - System Health: {health_score}/10",
#                 "data": result
#             }
#
#         except Exception as e:
#             return await self.on_error(e)
#
#     def _extract_metrics(self, input_data: dict) -> Dict[str, Any]:
#         metrics = {
#             "response_time": input_data.get("duration_ms", 0) / 1000,
#             "success": input_data.get("status") in ["completed", "success"],
#             "quality_score": input_data.get("quality_score", 0),
#             "agent_type": input_data.get("agent_type", "unknown"),
#             "timestamp": datetime.now()
#         }
#
#         # Store in history
#         self.metrics_history["response_times"].append(metrics["response_time"])
#         if len(self.metrics_history["response_times"]) > 1000:  # Keep last 1000
#             self.metrics_history["response_times"] = self.metrics_history["response_times"][-1000:]
#
#         return metrics
#
#     async def _analyze_trends(self, current_metrics: dict) -> Dict[str, Any]:
#         response_times = self.metrics_history["response_times"]
#
#         if len(response_times) < 2:
#             return {
#                 "trend": "insufficient_data",
#                 "anomalies": [],
#                 "statistics": {}
#             }
#
#         # Calculate basic statistics
#         avg_response = statistics.mean(response_times)
#         median_response = statistics.median(response_times)
#         std_dev = statistics.stdev(response_times) if len(response_times) > 1 else 0
#
#         # Detect anomalies (2 standard deviations from mean)
#         anomalies = []
#         if std_dev > 0:
#             z_score = abs(current_metrics["response_time"] - avg_response) / std_dev
#             if z_score > 2:
#                 anomalies.append(f"High response time anomaly detected: z-score {z_score:.2f}")
#
#         # Determine trend
#         recent_avg = statistics.mean(response_times[-10:]) if len(response_times) >= 10 else avg_response
#         trend = "improving" if recent_avg < avg_response else "degrading" if recent_avg > avg_response else "stable"
#
#         return {
#             "trend": trend,
#             "anomalies": anomalies,
#             "statistics": {
#                 "average_response_time": avg_response,
#                 "median_response_time": median_response,
#                 "standard_deviation": std_dev,
#                 "sample_size": len(response_times)
#             }
#         }
#
#     async def _generate_insights(self, trend_analysis: dict) -> Dict[str, List[str]]:
#         insights = []
#         recommendations = []
#
#         stats = trend_analysis["statistics"]
#
#         if stats.get("average_response_time", 0) > self.performance_thresholds["response_time"]:
#             insights.append("System response times are above optimal threshold")
#             recommendations.append("Consider optimizing database queries or scaling resources")
#
#         if trend_analysis["trend"] == "degrading":
#             insights.append("System performance is trending downward")
#             recommendations.append("Investigate recent changes or increased load patterns")
#
#         if trend_analysis["anomalies"]:
#             insights.append("Performance anomalies detected in recent activity")
#             recommendations.append("Review anomaly patterns for potential system issues")
#
#         if not insights:
#             insights.append("System performance within expected parameters")
#             recommendations.append("Continue monitoring for early detection of issues")
#
#         return {
#             "key_insights": insights,
#             "recommendations": recommendations
#         }
#
#     def _calculate_health_score(self, trend_analysis: dict) -> float:
#         base_score = 8.0  # Base healthy score
#
#         # Adjust based on trends
#         if trend_analysis["trend"] == "improving":
#             base_score += 1.0
#         elif trend_analysis["trend"] == "degrading":
#             base_score -= 2.0
#
#         # Adjust for anomalies
#         if trend_analysis["anomalies"]:
#             base_score -= len(trend_analysis["anomalies"]) * 0.5
#
#         # Ensure score stays in reasonable range
#         return max(0.0, min(10.0, base_score))


# /agents/a_series/analytics_agent.py
from core.agent_base import RevenantAgentBase
import asyncio
from typing import Dict, List, Any
from datetime import datetime
import statistics


class AnalyticsAgent(RevenantAgentBase):
    """
    Analyzes system performance, agent effectiveness, and provides data-driven insights.

    Input:
        - duration_ms (int): Execution duration in milliseconds
        - status (str): Execution status
        - quality_score (float): Quality score (0-10)
        - agent_type (str): Type of agent being analyzed

    Output:
        - system_health_score (float): Overall health score (0-10)
        - performance_trends (dict): Trend analysis results
        - key_insights (list): Important insights discovered
        - recommendations (list): Optimization recommendations
    """

    metadata = {
        "name": "AnalyticsAgent",
        "series": "a_series",
        "version": "0.1.0",
        "description": "Analyzes system performance, agent effectiveness, and provides data-driven insights for optimization."
    }

    def __init__(self):
        super().__init__(
            name="AnalyticsAgent",
            description="Analyzes system performance, agent effectiveness, and provides data-driven insights for optimization."
        )
        self.metrics_history = {}
        self.performance_thresholds = {
            "response_time": 5.0,  # seconds
            "success_rate": 0.95,  # 95%
            "quality_score": 7.0  # out of 10
        }
        self.metadata = AnalyticsAgent.metadata

    async def setup(self):
        # Initialize analytics storage and thresholds
        self.metrics_history = {
            "response_times": [],
            "success_rates": [],
            "quality_scores": [],
            "agent_performance": {}
        }
        await asyncio.sleep(0.1)

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Extract metrics from input
            metrics = self._extract_metrics(input_data)

            # Analyze performance trends
            trend_analysis = await self._analyze_trends(metrics)

            # Generate insights and recommendations
            insights = await self._generate_insights(trend_analysis)

            # Calculate system health score
            health_score = self._calculate_health_score(trend_analysis)

            result = {
                "system_health_score": health_score,
                "performance_trends": trend_analysis,
                "key_insights": insights["key_insights"],
                "recommendations": insights["recommendations"],
                "anomalies_detected": trend_analysis["anomalies"],
                "timestamp": datetime.now().isoformat()
            }

            return {
                "agent": self.name,
                "status": "ok",
                "summary": f"Analytics completed - System Health: {health_score}/10",
                "data": result
            }

        except Exception as e:
            return await self.on_error(e)

    def _extract_metrics(self, input_data: dict) -> Dict[str, Any]:
        metrics = {
            "response_time": input_data.get("duration_ms", 0) / 1000,
            "success": input_data.get("status") in ["completed", "success"],
            "quality_score": input_data.get("quality_score", 0),
            "agent_type": input_data.get("agent_type", "unknown"),
            "timestamp": datetime.now()
        }

        # Store in history
        self.metrics_history["response_times"].append(metrics["response_time"])
        if len(self.metrics_history["response_times"]) > 1000:  # Keep last 1000
            self.metrics_history["response_times"] = self.metrics_history["response_times"][-1000:]

        return metrics

    async def _analyze_trends(self, current_metrics: dict) -> Dict[str, Any]:
        response_times = self.metrics_history["response_times"]

        if len(response_times) < 2:
            return {
                "trend": "insufficient_data",
                "anomalies": [],
                "statistics": {}
            }

        # Calculate basic statistics
        avg_response = statistics.mean(response_times)
        median_response = statistics.median(response_times)
        std_dev = statistics.stdev(response_times) if len(response_times) > 1 else 0

        # Detect anomalies (2 standard deviations from mean)
        anomalies = []
        if std_dev > 0:
            z_score = abs(current_metrics["response_time"] - avg_response) / std_dev
            if z_score > 2:
                anomalies.append(f"High response time anomaly detected: z-score {z_score:.2f}")

        # Determine trend
        recent_avg = statistics.mean(response_times[-10:]) if len(response_times) >= 10 else avg_response
        trend = "improving" if recent_avg < avg_response else "degrading" if recent_avg > avg_response else "stable"

        return {
            "trend": trend,
            "anomalies": anomalies,
            "statistics": {
                "average_response_time": avg_response,
                "median_response_time": median_response,
                "standard_deviation": std_dev,
                "sample_size": len(response_times)
            }
        }

    async def _generate_insights(self, trend_analysis: dict) -> Dict[str, List[str]]:
        insights = []
        recommendations = []

        stats = trend_analysis["statistics"]

        if stats.get("average_response_time", 0) > self.performance_thresholds["response_time"]:
            insights.append("System response times are above optimal threshold")
            recommendations.append("Consider optimizing database queries or scaling resources")

        if trend_analysis["trend"] == "degrading":
            insights.append("System performance is trending downward")
            recommendations.append("Investigate recent changes or increased load patterns")

        if trend_analysis["anomalies"]:
            insights.append("Performance anomalies detected in recent activity")
            recommendations.append("Review anomaly patterns for potential system issues")

        if not insights:
            insights.append("System performance within expected parameters")
            recommendations.append("Continue monitoring for early detection of issues")

        return {
            "key_insights": insights,
            "recommendations": recommendations
        }

    def _calculate_health_score(self, trend_analysis: dict) -> float:
        base_score = 8.0  # Base healthy score

        # Adjust based on trends
        if trend_analysis["trend"] == "improving":
            base_score += 1.0
        elif trend_analysis["trend"] == "degrading":
            base_score -= 2.0

        # Adjust for anomalies
        if trend_analysis["anomalies"]:
            base_score -= len(trend_analysis["anomalies"]) * 0.5

        # Ensure score stays in reasonable range
        return max(0.0, min(10.0, base_score))
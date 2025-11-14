# agents/d_series/evaluator_agent.py
"""
Evaluator Agent for Revenant Framework
D-Series: Performance, Accuracy, and Consistency Evaluation
Provides quality assessment and conflict detection for multi-agent outputs.
"""

import logging
import statistics
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from core.agent_base import RevenantAgentBase

logger = logging.getLogger(__name__)


class EvaluatorAgent(RevenantAgentBase):
    """Evaluates agent performance, detects conflicts, and assigns confidence scores."""

    metadata = {
        "name": "LegalAgent",
        "version": "1.0.0",
        "series": "d_series",
        "description": "Performs performance and consistency evaluations for Revenant multi-agent workflows" ,
        "module": "agents.d_series.evaluator_agent"
    }
    def __init__(self):
        """Initialize EvaluatorAgent with scoring models and thresholds."""
        super().__init__()
        self.scoring_weights = {
            'completeness': 0.25,
            'consistency': 0.30,
            'confidence': 0.20,
            'timeliness': 0.15,
            'relevance': 0.10
        }
        self.conflict_threshold = 0.3
        self.low_confidence_threshold = 0.6
        logger.info(f"Initialized {self.metadata['series']}-Series EvaluatorAgent v{self.metadata['version']}")

    async def evaluate_outputs(self, agent_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Comprehensive evaluation of multiple agent outputs.

        Args:
            agent_results: List of agent output dictionaries

        Returns:
            Evaluation report with scores, conflicts, and recommendations
        """
        logger.info(f"Evaluating {len(agent_results)} agent outputs")

        try:
            # Individual agent evaluations
            individual_scores = []
            for result in agent_results:
                score_report = await self.compute_score(result)
                individual_scores.append(score_report)

            # Conflict detection
            conflicts = await self.detect_conflicts(agent_results)

            # Overall assessment
            overall_score = self._compute_overall_score(individual_scores)
            assessment_summary = await self.summarize_assessment({
                "individual_scores": individual_scores,
                "conflicts": conflicts,
                "overall_score": overall_score
            })

            evaluation_report = {
                "status": "completed",
                "evaluation_timestamp": datetime.utcnow().isoformat(),
                "overall_score": overall_score,
                "individual_scores": individual_scores,
                "conflicts_detected": conflicts,
                "risk_assessment": self._assess_risk(individual_scores, conflicts),
                "recommendations": self._generate_recommendations(individual_scores, conflicts),
                "summary": assessment_summary,
                "metrics": {
                    "agents_evaluated": len(agent_results),
                    "high_confidence_agents": len([s for s in individual_scores if s['composite_score'] >= 0.8]),
                    "conflict_count": len(conflicts),
                    "evaluation_duration": "instant"  # Would be actual duration in real implementation
                }
            }

            logger.info(f"Evaluation completed: {overall_score['composite']:.2f} composite score")
            return evaluation_report

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "overall_score": {"composite": 0.0},
                "individual_scores": [],
                "conflicts_detected": [],
                "summary": f"Evaluation failed: {str(e)}"
            }

    async def compute_score(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute comprehensive score for a single agent result.

        Args:
            result: Single agent output dictionary

        Returns:
            Detailed scoring report
        """
        try:
            scores = {}

            # Completeness score
            scores['completeness'] = self._score_completeness(result)

            # Consistency score
            scores['consistency'] = self._score_consistency(result)

            # Confidence score (from agent's own confidence if available)
            scores['confidence'] = self._score_confidence(result)

            # Timeliness score
            scores['timeliness'] = self._score_timeliness(result)

            # Relevance score
            scores['relevance'] = self._score_relevance(result)

            # Composite weighted score
            composite_score = sum(scores[metric] * weight
                                  for metric, weight in self.scoring_weights.items())

            scoring_report = {
                "agent_id": result.get('agent_id', 'unknown'),
                "composite_score": round(composite_score, 3),
                "component_scores": scores,
                "confidence_level": self._classify_confidence(composite_score),
                "timestamp": result.get('timestamp', datetime.utcnow().isoformat()),
                "evaluation_notes": self._generate_score_notes(scores, result)
            }

            logger.debug(f"Computed score for {scoring_report['agent_id']}: {composite_score:.3f}")
            return scoring_report

        except Exception as e:
            logger.error(f"Score computation failed for agent {result.get('agent_id', 'unknown')}: {str(e)}")
            return {
                "agent_id": result.get('agent_id', 'unknown'),
                "composite_score": 0.0,
                "component_scores": {},
                "confidence_level": "error",
                "error": str(e)
            }

    async def detect_conflicts(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect conflicts and inconsistencies across multiple agent outputs.

        Args:
            results: List of agent outputs to analyze for conflicts

        Returns:
            List of detected conflicts with details
        """
        conflicts = []

        if len(results) < 2:
            return conflicts

        try:
            # Extract key decision points from each result
            decision_points = self._extract_decision_points(results)

            for point_name, values in decision_points.items():
                if self._is_conflicting(values):
                    conflict = {
                        "conflict_id": f"conflict_{point_name}",
                        "point_name": point_name,
                        "conflicting_values": values,
                        "severity": self._assess_conflict_severity(values),
                        "agents_involved": [v['agent_id'] for v in values],
                        "resolution_suggestion": self._suggest_conflict_resolution(values)
                    }
                    conflicts.append(conflict)

            logger.info(f"Detected {len(conflicts)} conflicts across {len(results)} agent outputs")
            return conflicts

        except Exception as e:
            logger.error(f"Conflict detection failed: {str(e)}")
            return []

    async def summarize_assessment(self, report: Dict[str, Any]) -> str:
        """
        Generate human-readable evaluation summary.

        Args:
            report: Complete evaluation report

        Returns:
            Concise natural language summary
        """
        try:
            overall = report.get('overall_score', {})
            individual_scores = report.get('individual_scores', [])
            conflicts = report.get('conflicts_detected', [])

            if not individual_scores:
                return "No agent outputs to evaluate."

            # Calculate basic statistics
            composite_scores = [s['composite_score'] for s in individual_scores]
            avg_score = statistics.mean(composite_scores) if composite_scores else 0
            min_score = min(composite_scores) if composite_scores else 0
            max_score = max(composite_scores) if composite_scores else 0

            # Generate summary
            summary_parts = []

            summary_parts.append(
                f"Evaluated {len(individual_scores)} agents with average score: {avg_score:.2f} "
                f"(range: {min_score:.2f}-{max_score:.2f})"
            )

            if conflicts:
                summary_parts.append(f"Detected {len(conflicts)} conflicts requiring resolution")
            else:
                summary_parts.append("No significant conflicts detected")

            # Add risk assessment
            risk_level = report.get('risk_assessment', {}).get('level', 'unknown')
            summary_parts.append(f"Overall risk level: {risk_level.upper()}")

            # Add top performer if available
            if individual_scores:
                top_agent = max(individual_scores, key=lambda x: x['composite_score'])
                summary_parts.append(
                    f"Top performer: {top_agent['agent_id']} ({top_agent['composite_score']:.2f})"
                )

            return ". ".join(summary_parts)

        except Exception as e:
            logger.error(f"Assessment summarization failed: {str(e)}")
            return f"Summary generation failed: {str(e)}"

    def _score_completeness(self, result: Dict[str, Any]) -> float:
        """Score based on data completeness and structure."""
        data = result.get('data', {})

        if not data:
            return 0.0

        # Check for expected fields based on result type
        result_type = result.get('type', 'unknown')
        expected_fields = self._get_expected_fields(result_type)

        if not expected_fields:
            return 0.7  # Default moderate score for unknown types

        present_fields = [field for field in expected_fields if field in data]
        completeness_ratio = len(present_fields) / len(expected_fields)

        return completeness_ratio

    def _score_consistency(self, result: Dict[str, Any]) -> float:
        """Score internal consistency of the result."""
        data = result.get('data', {})

        # Check for internal contradictions
        contradictions = 0
        total_checks = 0

        # Example consistency checks
        if 'confidence' in result and 'data' in data:
            total_checks += 1
            # High confidence should correlate with detailed data
            if result['confidence'] > 0.8 and len(str(data.get('data', ''))) < 10:
                contradictions += 1

        # Add more consistency checks as needed

        consistency_score = 1.0 - (contradictions / max(total_checks, 1))
        return consistency_score

    def _score_confidence(self, result: Dict[str, Any]) -> float:
        """Score based on agent's self-reported confidence."""
        reported_confidence = result.get('confidence', 0.5)

        # Validate confidence score
        if not isinstance(reported_confidence, (int, float)):
            return 0.5

        # Normalize to 0-1 range if needed
        if reported_confidence > 1.0:
            reported_confidence = 1.0
        elif reported_confidence < 0.0:
            reported_confidence = 0.0

        return reported_confidence

    def _score_timeliness(self, result: Dict[str, Any]) -> float:
        """Score based on response timeliness."""
        timestamp = result.get('timestamp')
        if not timestamp:
            return 0.5

        try:
            # Calculate recency (simplified)
            # In real implementation, compare with task creation time
            return 0.8  # Default good score for now
        except:
            return 0.5

    def _score_relevance(self, result: Dict[str, Any]) -> float:
        """Score relevance to the original task."""
        # Check if result contains relevant keywords or matches expected output structure
        task_context = result.get('context', {})
        data = result.get('data', {})

        # Simple relevance check based on data presence and context alignment
        if data and task_context:
            return 0.7  # Moderate relevance

        return 0.3  # Low relevance

    def _compute_overall_score(self, individual_scores: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute overall evaluation scores."""
        if not individual_scores:
            return {"composite": 0.0}

        composite_scores = [s['composite_score'] for s in individual_scores]

        return {
            "composite": statistics.mean(composite_scores),
            "min": min(composite_scores),
            "max": max(composite_scores),
            "median": statistics.median(composite_scores),
            "std_dev": statistics.stdev(composite_scores) if len(composite_scores) > 1 else 0.0
        }

    def _extract_decision_points(self, results: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Extract key decision points from agent results."""
        decision_points = {}

        for result in results:
            agent_id = result.get('agent_id', f'agent_{len(decision_points)}')
            data = result.get('data', {})

            for key, value in data.items():
                if key not in decision_points:
                    decision_points[key] = []

                decision_points[key].append({
                    'agent_id': agent_id,
                    'value': value,
                    'confidence': result.get('confidence', 0.5)
                })

        return decision_points

    def _is_conflicting(self, values: List[Dict]) -> bool:
        """Determine if values represent a meaningful conflict."""
        if len(values) < 2:
            return False

        # Extract unique values
        unique_values = set()
        for value_info in values:
            # Convert to string for comparison, handle different types
            unique_values.add(str(value_info['value']))

        # Consider it a conflict if multiple distinct values
        return len(unique_values) > 1

    def _assess_conflict_severity(self, values: List[Dict]) -> str:
        """Assess severity of detected conflict."""
        value_diversity = len(set(str(v['value']) for v in values))
        avg_confidence = statistics.mean(v.get('confidence', 0.5) for v in values)

        if value_diversity > 2 and avg_confidence > 0.7:
            return "high"
        elif value_diversity == 2 and avg_confidence > 0.6:
            return "medium"
        else:
            return "low"

    def _suggest_conflict_resolution(self, values: List[Dict]) -> str:
        """Suggest resolution strategy for conflict."""
        confidences = [v.get('confidence', 0.5) for v in values]
        max_confidence = max(confidences)

        if max_confidence > 0.8:
            return "Use highest-confidence value"
        else:
            return "Requires manual review or FusionAgent intervention"

    def _assess_risk(self, scores: List[Dict[str, Any]], conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall risk based on scores and conflicts."""
        composite_scores = [s['composite_score'] for s in scores]
        avg_score = statistics.mean(composite_scores) if composite_scores else 0
        conflict_count = len(conflicts)

        if avg_score < 0.4 or conflict_count > 3:
            risk_level = "high"
        elif avg_score < 0.7 or conflict_count > 1:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "level": risk_level,
            "factors": {
                "low_scores": avg_score < 0.7,
                "conflicts_present": conflict_count > 0,
                "score_variance": statistics.stdev(composite_scores) if len(composite_scores) > 1 else 0.0
            }
        }

    def _generate_recommendations(self, scores: List[Dict[str, Any]],
                                  conflicts: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        low_scorers = [s for s in scores if s['composite_score'] < self.low_confidence_threshold]
        if low_scorers:
            recommendations.append(
                f"Review performance of {len(low_scorers)} low-scoring agents"
            )

        if conflicts:
            recommendations.append(
                f"Resolve {len(conflicts)} detected conflicts before proceeding"
            )

        # Check for score consistency
        composite_scores = [s['composite_score'] for s in scores]
        if len(composite_scores) > 1:
            score_range = max(composite_scores) - min(composite_scores)
            if score_range > 0.3:
                recommendations.append("High score variance detected - consider agent calibration")

        if not recommendations:
            recommendations.append("No immediate actions required - outputs are satisfactory")

        return recommendations

    def _classify_confidence(self, score: float) -> str:
        """Classify confidence level based on score."""
        if score >= 0.8:
            return "high"
        elif score >= 0.6:
            return "medium"
        else:
            return "low"

    def _generate_score_notes(self, scores: Dict[str, float], result: Dict[str, Any]) -> List[str]:
        """Generate explanatory notes for scoring."""
        notes = []

        if scores['completeness'] < 0.5:
            notes.append("Low completeness - missing expected data fields")

        if scores['consistency'] < 0.7:
            notes.append("Potential internal inconsistencies detected")

        if scores['confidence'] < 0.6:
            notes.append("Low self-reported confidence")

        return notes

    def _get_expected_fields(self, result_type: str) -> List[str]:
        """Get expected fields for different result types."""
        field_templates = {
            'analysis': ['insights', 'recommendations', 'confidence'],
            'action': ['action_taken', 'result', 'status'],
            'monitoring': ['metrics', 'alerts', 'status'],
            'security': ['threat_level', 'actions', 'recommendations']
        }

        return field_templates.get(result_type, [])
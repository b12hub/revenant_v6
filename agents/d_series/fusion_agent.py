# agents/d_series/fusion_agent.py
"""
Fusion Agent for Revenant Framework
D-Series: Cognitive Fusion and Conflict Resolution
Synthesizes multi-agent outputs into coherent decisions and insights.
"""

import logging
from typing import Dict, List, Any
from core.agent_base import RevenantAgentBase

logger = logging.getLogger(__name__)


class FusionAgent(RevenantAgentBase):
    """Fuses multiple agent outputs into unified, consistent decisions."""

    metadata = {
        "name": "FusionAgent",
        "version": "1.0.0",
        "series": "d_series",
        "description": "Multi-agent output fusion and conflict resolution" ,
        "module": "agents.d_series.fusion_agent"
    }
    def __init__(self):
        """Initialize FusionAgent with conflict resolution strategies."""
        super().__init__(    name=self.metadata['name'] ,
            description=self.metadata['description']
        )
        self.conflict_strategies = {
            'confidence_weighted': self._confidence_weighted_resolution,
            'majority_vote': self._majority_vote_resolution,
            'context_aware': self._context_aware_resolution,
            'temporal_recent': self._temporal_recent_resolution
        }
        logger.info(f"Initialized {self.metadata['series']}-Series FusionAgent v{self.metadata['version']}")

    async def fuse(self, agent_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fuse multiple agent outputs into coherent decision.

        Args:
            agent_outputs: List of agent output dictionaries with data and metadata

        Returns:
            Fused result with conflict resolution and confidence scoring
        """
        logger.info(f"Fusing {len(agent_outputs)} agent outputs")

        if not agent_outputs:
            return self._create_empty_fusion()

        if len(agent_outputs) == 1:
            return self._handle_single_output(agent_outputs[0])

        try:
            # Analyze outputs for conflicts and overlaps
            analysis = self._analyze_outputs(agent_outputs)

            # Select appropriate fusion strategy
            strategy = self._select_fusion_strategy(analysis)
            logger.debug(f"Selected fusion strategy: {strategy}")

            # Apply fusion
            fused_data = await self.conflict_strategies[strategy](agent_outputs, analysis)

            # Merge contexts
            merged_context = self.merge_contexts(agent_outputs)

            # Compute overall confidence
            overall_confidence = self.compute_confidence(fused_data)

            return {
                "status": "fused",
                "fused_data": fused_data,
                "context": merged_context,
                "confidence": overall_confidence,
                "fusion_strategy": strategy,
                "analysis": analysis,
                "sources_processed": len(agent_outputs),
                "timestamp": "2024-01-01T00:00:00Z"  # Should be actual timestamp
            }

        except Exception as e:
            logger.error(f"Fusion failed: {str(e)}")
            return self._create_error_fusion(str(e))

    def compute_confidence(self, data: Dict[str, Any]) -> float:
        """
        Compute confidence score for fused data.

        Args:
            data: Fused data dictionary

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence factors
        factors = []

        # Factor 1: Data completeness
        completeness = min(1.0, len(data) / 10.0)  # Normalize by expected fields
        factors.append(completeness * 0.3)

        # Factor 2: Value consistency (if we have multiple values for same field)
        if 'value_consistency' in data.get('metadata', {}):
            consistency = data['metadata']['value_consistency']
            factors.append(consistency * 0.4)

        # Factor 3: Source agreement (if available)
        if 'agreement_score' in data.get('metadata', {}):
            agreement = data['metadata']['agreement_score']
            factors.append(agreement * 0.3)

        # If no special factors, return moderate confidence
        if not factors:
            return 0.7

        return sum(factors) / len(factors)

    def merge_contexts(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge context information from multiple agent outputs.

        Args:
            data_list: List of agent outputs with context information

        Returns:
            Merged context dictionary
        """
        merged_context = {
            "sources": [],
            "timestamps": [],
            "environments": set(),
            "constraints": {}
        }

        for data in data_list:
            context = data.get('context', {})

            # Collect sources
            if 'agent_id' in data:
                merged_context["sources"].append(data['agent_id'])

            # Collect timestamps
            if 'timestamp' in data:
                merged_context["timestamps"].append(data['timestamp'])

            # Collect environments
            if 'environment' in context:
                merged_context["environments"].add(context['environment'])

            # Merge constraints (most restrictive wins)
            if 'constraints' in context:
                for key, value in context['constraints'].items():
                    if key not in merged_context["constraints"]:
                        merged_context["constraints"][key] = value
                    else:
                        # Take more restrictive constraint
                        current = merged_context["constraints"][key]
                        if isinstance(value, (int, float)) and isinstance(current, (int, float)):
                            merged_context["constraints"][key] = min(current, value)

        # Convert set to list for JSON serialization
        merged_context["environments"] = list(merged_context["environments"])

        return merged_context

    def _analyze_outputs(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze agent outputs for conflicts, overlaps, and patterns."""
        analysis = {
            "total_outputs": len(outputs),
            "conflicts": [],
            "overlaps": [],
            "confidence_stats": {},
            "consensus_level": "unknown"
        }

        # Extract confidence scores
        confidences = [out.get('confidence', 0.5) for out in outputs]
        analysis["confidence_stats"] = {
            "min": min(confidences),
            "max": max(confidences),
            "avg": sum(confidences) / len(confidences),
            "std": self._calculate_std(confidences)
        }

        # Detect conflicts in key fields
        key_fields = self._extract_key_fields(outputs)
        for field, values in key_fields.items():
            if len(set(values)) > 1:  # Multiple different values
                analysis["conflicts"].append({
                    "field": field,
                    "values": values,
                    "agents": [out.get('agent_id', f'agent_{i}')
                               for i, out in enumerate(outputs)]
                })

        # Detect overlapping fields
        all_fields = set()
        for output in outputs:
            all_fields.update(output.get('data', {}).keys())

        for field in all_fields:
            field_presence = [field in out.get('data', {}) for out in outputs]
            if sum(field_presence) > 1:
                analysis["overlaps"].append(field)

        # Determine consensus level
        conflict_count = len(analysis["conflicts"])
        if conflict_count == 0:
            analysis["consensus_level"] = "high"
        elif conflict_count < len(outputs) / 2:
            analysis["consensus_level"] = "medium"
        else:
            analysis["consensus_level"] = "low"

        return analysis

    def _select_fusion_strategy(self, analysis: Dict[str, Any]) -> str:
        """Select appropriate fusion strategy based on output analysis."""
        consensus = analysis.get("consensus_level", "unknown")
        confidence_std = analysis["confidence_stats"].get("std", 0)

        if confidence_std > 0.3:
            return 'confidence_weighted'
        elif consensus == 'low':
            return 'context_aware'
        elif len(analysis.get('conflicts', [])) > 0:
            return 'majority_vote'
        else:
            return 'confidence_weighted'

    async def _confidence_weighted_resolution(self, outputs: List[Dict[str, Any]],
                                              analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflicts using confidence-weighted averaging."""
        fused_data = {}

        # Group by fields
        all_fields = set()
        for output in outputs:
            all_fields.update(output.get('data', {}).keys())

        for field in all_fields:
            values = []
            weights = []

            for output in outputs:
                data = output.get('data', {})
                if field in data:
                    values.append(data[field])
                    weights.append(output.get('confidence', 0.5))

            if values:
                # For numeric values, use weighted average
                if all(isinstance(v, (int, float)) for v in values):
                    weighted_sum = sum(v * w for v, w in zip(values, weights))
                    total_weight = sum(weights)
                    fused_data[field] = weighted_sum / total_weight if total_weight > 0 else sum(values) / len(values)
                else:
                    # For non-numeric, use highest confidence value
                    max_idx = weights.index(max(weights))
                    fused_data[field] = values[max_idx]

        return fused_data

    async def _majority_vote_resolution(self, outputs: List[Dict[str, Any]],
                                        analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflicts using majority voting."""
        fused_data = {}

        for conflict in analysis.get('conflicts', []):
            field = conflict['field']
            values = conflict['values']

            # Count occurrences of each value
            value_counts = {}
            for value in values:
                value_counts[value] = value_counts.get(value, 0) + 1

            # Select value with highest count
            fused_data[field] = max(value_counts.items(), key=lambda x: x[1])[0]

        # Add non-conflicting fields
        for output in outputs:
            for field, value in output.get('data', {}).items():
                if field not in fused_data and field not in [c['field'] for c in analysis.get('conflicts', [])]:
                    fused_data[field] = value

        return fused_data

    async def _context_aware_resolution(self, outputs: List[Dict[str, Any]],
                                        analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Context-aware conflict resolution using environmental factors."""
        # This would incorporate context analysis in real implementation
        # For now, fall back to confidence-weighted approach
        return await self._confidence_weighted_resolution(outputs, analysis)

    async def _temporal_recent_resolution(self, outputs: List[Dict[str, Any]],
                                          analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Prefer more recent outputs in case of conflicts."""
        # Sort by timestamp (most recent first)
        sorted_outputs = sorted(outputs,
                                key=lambda x: x.get('timestamp', ''),
                                reverse=True)

        fused_data = {}

        # Take values from most recent outputs, skipping conflicts
        for output in sorted_outputs:
            for field, value in output.get('data', {}).items():
                if field not in fused_data:
                    fused_data[field] = value

        return fused_data

    def _extract_key_fields(self, outputs: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """Extract values for key fields across all outputs."""
        key_fields = {}

        # Identify common important fields
        common_fields = set()
        for output in outputs:
            common_fields.update(output.get('data', {}).keys())

        for field in common_fields:
            values = []
            for output in outputs:
                data = output.get('data', {})
                if field in data:
                    values.append(data[field])
            if values:
                key_fields[field] = values

        return key_fields

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation of values."""
        if len(values) <= 1:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    def _create_empty_fusion(self) -> Dict[str, Any]:
        """Create empty fusion result."""
        return {
            "status": "no_data",
            "fused_data": {},
            "context": {},
            "confidence": 0.0,
            "fusion_strategy": "none",
            "sources_processed": 0
        }

    def _handle_single_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Handle case with only one agent output."""
        return {
            "status": "single_source",
            "fused_data": output.get('data', {}),
            "context": output.get('context', {}),
            "confidence": output.get('confidence', 0.5),
            "fusion_strategy": "direct_pass",
            "sources_processed": 1,
            "original_agent": output.get('agent_id')
        }

    def _create_error_fusion(self, error_msg: str) -> Dict[str, Any]:
        """Create error fusion result."""
        return {
            "status": "error",
            "fused_data": {},
            "context": {},
            "confidence": 0.0,
            "fusion_strategy": "none",
            "error": error_msg,
            "sources_processed": 0
        }
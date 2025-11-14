# # test_d_series_orchestration.py
# """
# Test script for D-Series orchestration functionality.
# Validates CoordinatorAgent delegation and FusionAgent merging.
# """
#
# import asyncio
# import logging
# from agents.d_series.coordinator_agent import CoordinatorAgent
# from agents.d_series.fusion_agent import FusionAgent
#
# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
#
# async def test_d_series_orchestration():
#     """Test the complete D-Series orchestration flow."""
#     logger.info("Starting D-Series Orchestration Test")
#
#     # Initialize agents
#     fusion_agent = FusionAgent()
#     coordinator = CoordinatorAgent(registry={})  # Mock registry
#
#     # Test task
#     test_task = {
#         "type": "iot_security_analysis",
#         "task_id": "test_001",
#         "timestamp": "2024-01-01T00:00:00Z",
#         "requirements": ["A", "B"],
#         "data": {
#             "sensor_readings": [25.5, 26.1, 24.8],
#             "network_activity": "suspicious"
#         },
#         "action": "analyze_and_respond",
#         "context": {
#             "environment": "production",
#             "priority": "high"
#         }
#     }
#
#     logger.info("1. Testing CoordinatorAgent task delegation...")
#     coordination_result = await coordinator.collaborate(test_task)
#
#     print("✓ CoordinatorAgent Result:")
#     print(f"  Task ID: {coordination_result['task_id']}")
#     print(f"  Status: {coordination_result['status']}")
#     print(f"  Agents Used: {coordination_result['metadata']['agents_used']}")
#     print(f"  Summary: {coordination_result['summary']}")
#
#     # Simulate multiple agent outputs for fusion
#     logger.info("2. Testing FusionAgent output merging...")
#     mock_agent_outputs = [
#         {
#             "agent_id": "A_analysis_agent",
#             "data": {"threat_level": "medium", "recommendation": "investigate"},
#             "confidence": 0.8,
#             "timestamp": "2024-01-01T00:00:01Z",
#             "context": {"analysis_type": "behavioral"}
#         },
#         {
#             "agent_id": "B_security_agent",
#             "data": {"threat_level": "high", "action_taken": "alert_sent"},
#             "confidence": 0.9,
#             "timestamp": "2024-01-01T00:00:02Z",
#             "context": {"response_type": "immediate"}
#         },
#         {
#             "agent_id": "A_monitoring_agent",
#             "data": {"threat_level": "low", "trend": "stable"},
#             "confidence": 0.6,
#             "timestamp": "2024-01-01T00:00:03Z",
#             "context": {"monitoring_scope": "comprehensive"}
#         }
#     ]
#
#     fusion_result = await fusion_agent.fuse(mock_agent_outputs)
#
#     print("✓ FusionAgent Result:")
#     print(f"  Status: {fusion_result['status']}")
#     print(f"  Strategy: {fusion_result['fusion_strategy']}")
#     print(f"  Confidence: {fusion_result['confidence']:.2f}")
#     print(f"  Fused Data: {fusion_result['fused_data']}")
#     print(f"  Sources: {fusion_result['sources_processed']}")
#
#     # Validate unified output structure
#     logger.info("3. Validating unified output JSON structure...")
#
#     unified_output = {
#         "orchestration_result": coordination_result,
#         "fusion_result": fusion_result,
#         "test_validation": {
#             "coordinator_success": coordination_result['status'] == 'completed',
#             "fusion_success": fusion_result['status'] in ['fused', 'single_source'],
#             "has_confidence": 'confidence' in fusion_result,
#             "has_sources": fusion_result['sources_processed'] > 0
#         }
#     }
#
#     print("✓ Unified Output Structure:")
#     print(f"  Coordinator Success: {unified_output['test_validation']['coordinator_success']}")
#     print(f"  Fusion Success: {unified_output['test_validation']['fusion_success']}")
#     print(f"  Has Confidence: {unified_output['test_validation']['has_confidence']}")
#     print(f"  Has Sources: {unified_output['test_validation']['has_sources']}")
#
#     logger.info("D-Series Orchestration Test Completed Successfully!")
#     return unified_output
#
#
# if __name__ == "__main__":
#     result = asyncio.run(test_d_series_orchestration())
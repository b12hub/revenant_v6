#!/usr/bin/env python3
"""
Test Suite for B-Series Phase 4 Agents
Tests IoTAgent and ARVRAgent functionality
"""

import asyncio
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AgentTest")


class AgentTestSuite:
    """Comprehensive test suite for IoT and AR/VR agents"""

    def __init__(self):
        self.iot_agent = None
        self.arvr_agent = None
        self.test_results = {
            "iot_agent": {"passed": 0, "failed": 0, "tests": []},
            "arvr_agent": {"passed": 0, "failed": 0, "tests": []}
        }

    async def setup_agents(self):
        """Initialize both agents"""
        try:
            from iot_agent import IoTAgent
            self.iot_agent = IoTAgent()
            logger.info("✅ IoTAgent initialized successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import IoTAgent: {e}")
            return False

        try:
            from arvr_agent import ARVRAgent
            self.arvr_agent = ARVRAgent()
            logger.info("✅ ARVRAgent initialized successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import ARVRAgent: {e}")
            return False

        return True

    def _record_test_result(self, agent_name: str, test_name: str, passed: bool, details: str = ""):
        """Record individual test results"""
        result = {
            "test": test_name,
            "passed": passed,
            "details": details,
            "timestamp": time.time()
        }

        self.test_results[agent_name]["tests"].append(result)
        if passed:
            self.test_results[agent_name]["passed"] += 1
        else:
            self.test_results[agent_name]["failed"] += 1

        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {status}: {test_name} - {details}")

    async def test_iot_basic_functionality(self):
        """Test basic IoT agent functionality"""
        logger.info("\n" + "=" * 50)
        logger.info("Testing IoT Agent Basic Functionality")
        logger.info("=" * 50)

        if not self.iot_agent:
            self._record_test_result("iot_agent", "Agent Initialization", False, "Agent not initialized")
            return

        try:
            # Test 1: Basic run
            result = await self.iot_agent.run()
            self._record_test_result("iot_agent", "Basic Execution", True, "Agent executed successfully")

            # Test 2: Response structure
            required_keys = ["status", "devices", "network", "summary"]
            if all(key in result for key in required_keys):
                self._record_test_result("iot_agent", "Response Structure", True, "All required keys present")
            else:
                missing = [key for key in required_keys if key not in result]
                self._record_test_result("iot_agent", "Response Structure", False, f"Missing keys: {missing}")

            # Test 3: Device data structure
            if result["devices"] and len(result["devices"]) > 0:
                device = result["devices"][0]
                device_keys = ["id", "type", "value", "status"]
                if all(key in device for key in device_keys):
                    self._record_test_result("iot_agent", "Device Structure", True, "Device data complete")
                else:
                    self._record_test_result("iot_agent", "Device Structure", False, "Incomplete device data")
            else:
                self._record_test_result("iot_agent", "Device Structure", False, "No devices returned")

            # Test 4: Network stats
            network_keys = ["latency_ms", "packet_loss", "uptime", "connected_devices"]
            if all(key in result["network"] for key in network_keys):
                self._record_test_result("iot_agent", "Network Stats", True, "Network diagnostics complete")
            else:
                self._record_test_result("iot_agent", "Network Stats", False, "Incomplete network data")

            # Test 5: Simulation mode detection
            if "simulation_mode" in result:
                mode = "simulation" if result["simulation_mode"] else "MQTT"
                self._record_test_result("iot_agent", "Mode Detection", True, f"Running in {mode} mode")
            else:
                self._record_test_result("iot_agent", "Mode Detection", False, "No mode information")

        except Exception as e:
            self._record_test_result("iot_agent", "Basic Execution", False, f"Exception: {str(e)}")

    async def test_iot_advanced_features(self):
        """Test advanced IoT agent features"""
        logger.info("\n" + "=" * 50)
        logger.info("Testing IoT Agent Advanced Features")
        logger.info("=" * 50)

        if not self.iot_agent:
            return

        try:
            # Test 1: Device lookup by ID
            test_device_id = "sensor_temp_1_1"
            device = self.iot_agent.get_device_by_id(test_device_id)
            if device:
                self._record_test_result("iot_agent", "Device Lookup by ID", True, f"Found device: {device.id}")
            else:
                self._record_test_result("iot_agent", "Device Lookup by ID", False,
                                         f"Device not found: {test_device_id}")

            # Test 2: Device lookup by type
            temp_devices = self.iot_agent.get_devices_by_type("temperature")
            if temp_devices:
                self._record_test_result("iot_agent", "Device Lookup by Type", True,
                                         f"Found {len(temp_devices)} temperature devices")
            else:
                self._record_test_result("iot_agent", "Device Lookup by Type", False, "No temperature devices found")

            # Test 3: Multiple executions (data should change)
            result1 = await self.iot_agent.run()
            await asyncio.sleep(1.1)  # Wait for next simulation cycle
            result2 = await self.iot_agent.run()

            # Check if sensor values changed (they should in simulation)
            if result1["devices"][0]["value"] != result2["devices"][0]["value"]:
                self._record_test_result("iot_agent", "Data Simulation", True, "Sensor values updated between runs")
            else:
                self._record_test_result("iot_agent", "Data Simulation", False, "Sensor values static")

            # Test 4: Device status monitoring
            active_count = len([d for d in result2["devices"] if d["status"] == "active"])
            if active_count > 0:
                self._record_test_result("iot_agent", "Status Monitoring", True, f"{active_count} active devices")
            else:
                self._record_test_result("iot_agent", "Status Monitoring", False, "No active devices")

        except Exception as e:
            self._record_test_result("iot_agent", "Advanced Features", False, f"Exception: {str(e)}")

    async def test_arvr_basic_functionality(self):
        """Test basic AR/VR agent functionality"""
        logger.info("\n" + "=" * 50)
        logger.info("Testing AR/VR Agent Basic Functionality")
        logger.info("=" * 50)

        if not self.arvr_agent:
            self._record_test_result("arvr_agent", "Agent Initialization", False, "Agent not initialized")
            return

        try:
            # Test 1: Basic scene generation
            result = await self.arvr_agent.run(scene_type="indoor", object_count=5)
            self._record_test_result("arvr_agent", "Basic Execution", True, "Scene generated successfully")

            # Test 2: Response structure
            required_keys = ["status", "scene", "metadata"]
            if all(key in result for key in required_keys):
                self._record_test_result("arvr_agent", "Response Structure", True, "All required keys present")
            else:
                missing = [key for key in required_keys if key not in result]
                self._record_test_result("arvr_agent", "Response Structure", False, f"Missing keys: {missing}")

            # Test 3: Scene structure
            scene_keys = ["objects", "lighting", "optimization", "lighting_analysis"]
            if all(key in result["scene"] for key in scene_keys):
                self._record_test_result("arvr_agent", "Scene Structure", True, "Scene data complete")
            else:
                self._record_test_result("arvr_agent", "Scene Structure", False, "Incomplete scene data")

            # Test 4: Object generation
            if len(result["scene"]["objects"]) == 5:
                self._record_test_result("arvr_agent", "Object Count", True, "Correct number of objects generated")
            else:
                self._record_test_result("arvr_agent", "Object Count", False,
                                         f"Expected 5 objects, got {len(result['scene']['objects'])}")

            # Test 5: Optimization data
            opt_keys = ["polycount", "lod_enabled", "object_count", "optimization_score"]
            if all(key in result["scene"]["optimization"] for key in opt_keys):
                self._record_test_result("arvr_agent", "Optimization Data", True, "Optimization data complete")
            else:
                self._record_test_result("arvr_agent", "Optimization Data", False, "Incomplete optimization data")

        except Exception as e:
            self._record_test_result("arvr_agent", "Basic Execution", False, f"Exception: {str(e)}")

    async def test_arvr_scene_types(self):
        """Test different scene types"""
        logger.info("\n" + "=" * 50)
        logger.info("Testing AR/VR Agent Scene Types")
        logger.info("=" * 50)

        if not self.arvr_agent:
            return

        scene_types = ["indoor", "outdoor", "sci-fi", "fantasy"]

        for scene_type in scene_types:
            try:
                result = await self.arvr_agent.run(scene_type=scene_type, object_count=3, seed=42)

                if result["status"] == "success" and result["scene"]["type"] == scene_type:
                    self._record_test_result("arvr_agent", f"Scene Type: {scene_type}", True,
                                             f"Generated {len(result['scene']['objects'])} objects")
                else:
                    self._record_test_result("arvr_agent", f"Scene Type: {scene_type}", False,
                                             f"Failed to generate {scene_type} scene")

            except Exception as e:
                self._record_test_result("arvr_agent", f"Scene Type: {scene_type}", False, f"Exception: {str(e)}")

    async def test_arvr_deterministic_generation(self):
        """Test deterministic scene generation with seeds"""
        logger.info("\n" + "=" * 50)
        logger.info("Testing AR/VR Agent Deterministic Generation")
        logger.info("=" * 50)

        if not self.arvr_agent:
            return

        try:
            # Generate two scenes with same seed - should be identical
            result1 = await self.arvr_agent.run(scene_type="sci-fi", object_count=4, seed=123)
            result2 = await self.arvr_agent.run(scene_type="sci-fi", object_count=4, seed=123)

            # Compare object positions (should be identical with same seed)
            objects1 = result1["scene"]["objects"]
            objects2 = result2["scene"]["objects"]

            all_positions_match = all(
                obj1["position"] == obj2["position"]
                for obj1, obj2 in zip(objects1, objects2)
            )

            if all_positions_match:
                self._record_test_result("arvr_agent", "Deterministic Generation", True,
                                         "Identical scenes generated with same seed")
            else:
                self._record_test_result("arvr_agent", "Deterministic Generation", False,
                                         "Scenes differ despite same seed")

            # Test different seeds produce different results
            result3 = await self.arvr_agent.run(scene_type="sci-fi", object_count=4, seed=456)
            objects3 = result3["scene"]["objects"]

            positions_differ = any(
                obj1["position"] != obj3["position"]
                for obj1, obj3 in zip(objects1, objects3)
            )

            if positions_differ:
                self._record_test_result("arvr_agent", "Seed Variation", True,
                                         "Different seeds produce different scenes")
            else:
                self._record_test_result("arvr_agent", "Seed Variation", False,
                                         "Different seeds produced identical scenes")

        except Exception as e:
            self._record_test_result("arvr_agent", "Deterministic Generation", False, f"Exception: {str(e)}")

    async def test_arvr_export_functionality(self):
        """Test scene export functionality"""
        logger.info("\n" + "=" * 50)
        logger.info("Testing AR/VR Agent Export Functionality")
        logger.info("=" * 50)

        if not self.arvr_agent:
            return

        try:
            # Generate a scene first
            scene_result = await self.arvr_agent.run(scene_type="indoor", object_count=3)

            # Test JSON export
            export_result = await self.arvr_agent.export_scene(scene_result, "json")

            if export_result["status"] == "success":
                self._record_test_result("arvr_agent", "JSON Export", True,
                                         f"Export successful - {export_result.get('file_size_estimate', 'N/A')}")
            else:
                self._record_test_result("arvr_agent", "JSON Export", False,
                                         f"Export failed: {export_result.get('message', 'Unknown error')}")

            # Test unsupported format
            invalid_export = await self.arvr_agent.export_scene(scene_result, "invalid_format")
            if invalid_export["status"] == "error":
                self._record_test_result("arvr_agent", "Invalid Format Handling", True,
                                         "Properly rejected invalid format")
            else:
                self._record_test_result("arvr_agent", "Invalid Format Handling", False,
                                         "Failed to reject invalid format")

        except Exception as e:
            self._record_test_result("arvr_agent", "Export Functionality", False, f"Exception: {str(e)}")

    async def test_performance(self):
        """Test agent performance"""
        logger.info("\n" + "=" * 50)
        logger.info("Testing Agent Performance")
        logger.info("=" * 50)

        # Test IoT Agent performance
        if self.iot_agent:
            start_time = time.time()
            for _ in range(5):
                await self.iot_agent.run()
            iot_duration = time.time() - start_time

            if iot_duration < 10:  # Should be relatively fast
                self._record_test_result("iot_agent", "Performance", True,
                                         f"5 executions in {iot_duration:.2f}s")
            else:
                self._record_test_result("iot_agent", "Performance", False,
                                         f"5 executions took too long: {iot_duration:.2f}s")

        # Test AR/VR Agent performance
        if self.arvr_agent:
            start_time = time.time()
            for _ in range(3):
                await self.arvr_agent.run(scene_type="outdoor", object_count=10)
            arvr_duration = time.time() - start_time

            if arvr_duration < 5:  # Should be fast
                self._record_test_result("arvr_agent", "Performance", True,
                                         f"3 scene generations in {arvr_duration:.2f}s")
            else:
                self._record_test_result("arvr_agent", "Performance", False,
                                         f"3 scene generations took too long: {arvr_duration:.2f}s")

    def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("\n" + "=" * 70)
        logger.info("TEST REPORT SUMMARY")
        logger.info("=" * 70)

        for agent_name, results in self.test_results.items():
            total_tests = results["passed"] + results["failed"]
            if total_tests > 0:
                success_rate = (results["passed"] / total_tests) * 100
            else:
                success_rate = 0

            logger.info(
                f"\n{agent_name.upper():<15} | Passed: {results['passed']:2d} | Failed: {results['failed']:2d} | Success: {success_rate:6.1f}%")

            # Show failed tests
            failed_tests = [test for test in results["tests"] if not test["passed"]]
            if failed_tests:
                logger.info("  Failed Tests:")
                for test in failed_tests:
                    logger.info(f"    - {test['test']}: {test['details']}")

        # Overall result
        total_passed = sum(r["passed"] for r in self.test_results.values())
        total_failed = sum(r["failed"] for r in self.test_results.values())
        total_tests = total_passed + total_failed

        logger.info("\n" + "=" * 70)
        if total_failed == 0:
            logger.info("🎉 ALL TESTS PASSED!")
        else:
            logger.info(f"OVERALL: {total_passed}/{total_tests} tests passed ({total_passed / total_tests * 100:.1f}%)")
        logger.info("=" * 70)

        return self.test_results

    async def run_all_tests(self):
        """Run complete test suite"""
        logger.info("Starting B-Series Phase 4 Agent Test Suite")
        logger.info("Initializing agents...")

        if not await self.setup_agents():
            logger.error("❌ Failed to initialize agents. Exiting.")
            return self.test_results

        # Run IoT Agent tests
        await self.test_iot_basic_functionality()
        await self.test_iot_advanced_features()

        # Run AR/VR Agent tests
        await self.test_arvr_basic_functionality()
        await self.test_arvr_scene_types()
        await self.test_arvr_deterministic_generation()
        await self.test_arvr_export_functionality()

        # Performance tests
        await self.test_performance()

        # Generate report
        return self.generate_test_report()


# Quick individual agent test functions
async def quick_test_iot_agent():
    """Quick test for IoT Agent only"""
    from agents.b_series.iot_agent import IoTAgent

    logger.info("🔧 Quick Testing IoT Agent...")
    agent = IoTAgent()

    # Test basic functionality
    result = await agent.run()
    print(f"Status: {result['status']}")
    print(f"Devices: {len(result['devices'])}")
    print(f"Simulation Mode: {result['simulation_mode']}")
    print(f"Active Devices: {result['summary']['active_devices']}")

    # Test device lookup
    device = agent.get_device_by_id("sensor_temp_1_1")
    if device:
        print(f"Found device: {device.id} - {device.type}: {device.value}")

    return result


async def quick_test_arvr_agent():
    """Quick test for AR/VR Agent only"""
    global result
    from agents.b_series.arvr_agent import ARVRAgent

    logger.info("👓 Quick Testing AR/VR Agent...")
    agent = ARVRAgent()

    # Test different scene types
    scenes = ["indoor", "sci-fi", "fantasy"]
    for scene_type in scenes:
        result = await agent.run(scene_type=scene_type, object_count=3)
        print(f"\n{scene_type.upper()} Scene:")
        print(f"  Status: {result['status']}")
        print(f"  Objects: {len(result['scene']['objects'])}")
        print(f"  Polycount: {result['scene']['optimization']['polycount']}")
        print(f"  LOD: {result['scene']['optimization']['lod_enabled']}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test B-Series Phase 4 Agents")
    parser.add_argument("--quick", action="store_true", help="Run quick individual tests")
    parser.add_argument("--iot", action="store_true", help="Test IoT Agent only")
    parser.add_argument("--arvr", action="store_true", help="Test AR/VR Agent only")

    args = parser.parse_args()

    if args.quick:
        if args.iot:
            asyncio.run(quick_test_iot_agent())
        elif args.arvr:
            asyncio.run(quick_test_arvr_agent())
        else:
            # Run both quick tests
            asyncio.run(quick_test_iot_agent())
            asyncio.run(quick_test_arvr_agent())
    else:
        # Run full test suite
        test_suite = AgentTestSuite()
        asyncio.run(test_suite.run_all_tests())

# test_agents.py
# import asyncio
# from agents.iot_agent import IoTAgent
# from agents.arvr_agent import ARVRAgent
#
# async def main():
#     iot = IoTAgent()
#     arvr = ARVRAgent()
#     print(await iot.run({"device_count": 3}))
#     print(await arvr.run({"scene_type": "industrial"}))
#
# asyncio.run(main())

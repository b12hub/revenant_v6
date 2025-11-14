# /agents/b_series/iot_agent.py
from core.agent_base import RevenantAgentBase
import asyncio
from typing import Dict, List, Any
from datetime import datetime, timedelta
import random
import statistics

class IoTAgent(RevenantAgentBase):
    """Simulate sensor readings, monitor IoT device data, and manage device communications."""
    metadata = {
        "name": "IoTAgent",
        "version": "1.0.0",
        "series": "b_series",
        "description": "Manages IoT device networks, simulates sensor data, and monitors device health and performance",
        "module": "agents.b_series.iot_agent"
    }
    def __init__(self):
        super().__init__(name=self.metadata["name"],
            description=self.metadata["description"])
        self.device_registry = {}
        self.sensor_types = {}

    async def setup(self):
        # Initialize IoT device types and sensor configurations
        self.sensor_types = {
            "temperature": {"unit": "°C", "range": (-20, 50), "normal_range": (18, 25)},
            "humidity": {"unit": "%", "range": (0, 100), "normal_range": (30, 60)},
            "pressure": {"unit": "hPa", "range": (900, 1100), "normal_range": (1000, 1020)},
            "motion": {"unit": "boolean", "range": (0, 1), "normal_range": (0, 0)},
            "light": {"unit": "lux", "range": (0, 1000), "normal_range": (100, 500)},
            "air_quality": {"unit": "ppm", "range": (0, 1000), "normal_range": (0, 50)}
        }

        # Initialize device registry
        self.device_registry = {
            "sensor_nodes": [],
            "gateways": [],
            "actuators": []
        }

        await asyncio.sleep(0.1)

    async def run(self, input_data: dict):
        try:
            # Validate input
            devices = input_data.get("devices", [])
            operation = input_data.get("operation", "monitor")

            if not devices:
                # Generate mock devices if none provided
                devices = await self._generate_mock_devices(5)

            # Perform requested operation
            if operation == "monitor":
                result = await self._monitor_devices(devices)
            elif operation == "simulate":
                result = await self._simulate_sensor_data(devices)
            elif operation == "analyze":
                result = await self._analyze_device_network(devices)
            else:
                result = await self._monitor_devices(devices)

            # Generate device health report
            health_report = await self._generate_health_report(devices, result)

            final_result = {
                "operation_performed": operation,
                "devices_processed": len(devices),
                "device_analysis": result,
                "health_report": health_report,
                "network_status": await self._assess_network_status(devices),
                "timestamp": datetime.now().isoformat(),
                "recommendations": await self._generate_iot_recommendations(devices, health_report)
            }

            return {
                "agent": self.name,
                "status": "ok",
                "summary": f"IoT operation '{operation}' completed: {len(devices)} devices processed, {health_report['overall_health']} health status",
                "data": final_result
            }

        except Exception as e:
            return await self.on_error(str(e))

    async def _generate_mock_devices(self, count: int) -> List[Dict[str, Any]]:
        """Generate mock IoT devices for demonstration"""
        device_types = ["sensor", "actuator", "gateway"]
        sensor_types = list(self.sensor_types.keys())

        devices = []
        for i in range(count):
            device_type = random.choice(device_types)
            device = {
                "id": f"device_{i:03d}",
                "type": device_type,
                "name": f"{device_type.capitalize()} Node {i}",
                "location": f"Area {random.randint(1, 5)}",
                "status": "online",
                "battery_level": random.randint(20, 100),
                "last_seen": datetime.now() - timedelta(minutes=random.randint(0, 60)),
                "firmware_version": f"1.{random.randint(0, 5)}.{random.randint(0, 9)}"
            }

            if device_type == "sensor":
                device["sensor_type"] = random.choice(sensor_types)
                device["current_value"] = await self._generate_sensor_reading(device["sensor_type"])
                device["sensor_config"] = self.sensor_types.get(device["sensor_type"], {})

            elif device_type == "actuator":
                device["actuator_type"] = random.choice(["relay", "servo", "motor", "valve"])
                device["current_state"] = random.choice(["on", "off", "idle"])

            elif device_type == "gateway":
                device["connected_devices"] = random.randint(1, 10)
                device["network_type"] = random.choice(["wifi", "lora", "cellular", "ethernet"])

            devices.append(device)

        return devices

    async def _generate_sensor_reading(self, sensor_type: str) -> float:
        """Generate realistic sensor reading based on sensor type"""
        config = self.sensor_types.get(sensor_type, {})
        range_min, range_max = config.get("range", (0, 100))

        if sensor_type == "temperature":
            # Simulate temperature with some variation
            base_temp = 22.0
            variation = random.uniform(-5, 5)
            return round(base_temp + variation, 1)

        elif sensor_type == "humidity":
            base_humidity = 45.0
            variation = random.uniform(-15, 15)
            return round(max(0, min(100, base_humidity + variation)), 1)

        elif sensor_type == "pressure":
            base_pressure = 1013.0
            variation = random.uniform(-10, 10)
            return round(base_pressure + variation, 1)

        elif sensor_type == "motion":
            return 1.0 if random.random() > 0.8 else 0.0

        elif sensor_type == "light":
            # Simulate day/night cycle
            current_hour = datetime.now().hour
            if 6 <= current_hour <= 18:  # Daytime
                base_light = 300
                variation = random.uniform(-100, 200)
            else:  # Nighttime
                base_light = 50
                variation = random.uniform(-30, 50)
            return round(max(0, base_light + variation), 1)

        elif sensor_type == "air_quality":
            base_quality = 25.0
            variation = random.uniform(-10, 30)
            return round(max(0, base_quality + variation), 1)

        else:
            return round(random.uniform(range_min, range_max), 2)

    async def _monitor_devices(self, devices: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Monitor current device status and readings"""
        online_devices = []
        offline_devices = []
        sensor_readings = {}
        device_health = {}

        for device in devices:
            # Check device status (simulate occasional offline devices)
            is_online = await self._check_device_online(device)

            if is_online:
                online_devices.append(device["id"])

                # Update sensor readings for online sensors
                if device.get("sensor_type"):
                    current_value = await self._generate_sensor_reading(device["sensor_type"])
                    device["current_value"] = current_value

                    # Store reading by sensor type
                    sensor_type = device["sensor_type"]
                    if sensor_type not in sensor_readings:
                        sensor_readings[sensor_type] = []
                    sensor_readings[sensor_type].append(current_value)

                # Assess device health
                health_status = await self._assess_device_health(device)
                device_health[device["id"]] = health_status
            else:
                offline_devices.append(device["id"])
                device["status"] = "offline"

        return {
            "online_devices": online_devices,
            "offline_devices": offline_devices,
            "online_rate": len(online_devices) / len(devices) if devices else 0,
            "sensor_readings": sensor_readings,
            "device_health": device_health,
            "current_timestamp": datetime.now().isoformat(),
            "readings_summary": await self._summarize_readings(sensor_readings)
        }

    async def _simulate_sensor_data(self, devices: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simulate sensor data stream and generate historical data"""
        sensor_devices = [d for d in devices if d.get("sensor_type")]
        simulated_data = {}
        historical_trends = {}

        for device in sensor_devices:
            sensor_type = device["sensor_type"]
            device_id = device["id"]

            # Generate time series data (last 24 hours, hourly)
            historical_readings = []
            for hours_ago in range(24, 0, -1):
                timestamp = datetime.now() - timedelta(hours=hours_ago)
                reading = await self._generate_historical_reading(sensor_type, hours_ago)
                historical_readings.append({
                    "timestamp": timestamp.isoformat(),
                    "value": reading
                })

            # Current reading
            current_reading = await self._generate_sensor_reading(sensor_type)

            simulated_data[device_id] = {
                "sensor_type": sensor_type,
                "current_reading": current_reading,
                "historical_data": historical_readings,
                "data_quality": await self._assess_data_quality(historical_readings, current_reading),
                "anomalies": await self._detect_anomalies(historical_readings, current_reading)
            }

            # Analyze trends
            historical_trends[device_id] = await self._analyze_trends(historical_readings)

        return {
            "simulated_devices": len(sensor_devices),
            "sensor_data": simulated_data,
            "trend_analysis": historical_trends,
            "data_quality_report": await self._generate_data_quality_report(simulated_data),
            "simulation_timestamp": datetime.now().isoformat()
        }

    async def _analyze_device_network(self, devices: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze IoT device network topology and performance"""
        gateways = [d for d in devices if d.get("type") == "gateway"]
        sensors = [d for d in devices if d.get("type") == "sensor"]
        actuators = [d for d in devices if d.get("type") == "actuator"]

        return {
            "network_topology": {
                "total_devices": len(devices),
                "gateways": len(gateways),
                "sensors": len(sensors),
                "actuators": len(actuators),
                "device_distribution": await self._analyze_device_distribution(devices)
            },
            "performance_metrics": {
                "average_battery": await self._calculate_average_battery(devices),
                "network_coverage": await self._assess_network_coverage(devices),
                "data_throughput": await self._estimate_data_throughput(devices),
                "latency_estimate": await self._estimate_network_latency(devices)
            },
            "security_assessment": await self._assess_network_security(devices),
            "optimization_opportunities": await self._identify_optimization_opportunities(devices)
        }

    async def _generate_health_report(self, devices: List[Dict[str, Any]], analysis_result: Dict[str, Any]) -> Dict[
        str, Any]:
        """Generate comprehensive device health report"""
        device_health = analysis_result.get("device_health", {})
        online_rate = analysis_result.get("online_rate", 0)

        health_scores = []
        for device_id, health in device_health.items():
            health_scores.append(health.get("score", 0))

        avg_health = statistics.mean(health_scores) if health_scores else 0

        # Determine overall health status
        if avg_health >= 80 and online_rate >= 0.9:
            overall_health = "excellent"
        elif avg_health >= 60 and online_rate >= 0.7:
            overall_health = "good"
        elif avg_health >= 40 and online_rate >= 0.5:
            overall_health = "fair"
        else:
            overall_health = "poor"

        return {
            "overall_health": overall_health,
            "average_health_score": round(avg_health, 1),
            "online_rate": online_rate,
            "critical_issues": await self._identify_critical_issues(devices, device_health),
            "maintenance_recommendations": await self._generate_maintenance_recommendations(devices, device_health),
            "battery_status": await self._analyze_battery_status(devices)
        }

    async def _assess_network_status(self, devices: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall network status"""
        online_devices = [d for d in devices if d.get("status") == "online"]
        gateway_devices = [d for d in devices if d.get("type") == "gateway"]

        return {
            "network_status": "stable" if len(online_devices) / len(devices) > 0.8 else "degraded",
            "gateway_coverage": len(gateway_devices) > 0,
            "device_connectivity": len(online_devices),
            "network_load": await self._calculate_network_load(devices),
            "recommended_actions": await self._suggest_network_improvements(devices)
        }

    # Helper methods with mock implementations
    async def _check_device_online(self, device: Dict[str, Any]) -> bool:
        """Check if device is online (mock implementation)"""
        # Simulate 90% online rate
        return random.random() > 0.1

    async def _assess_device_health(self, device: Dict[str, Any]) -> Dict[str, Any]:
        """Assess individual device health"""
        battery_level = device.get("battery_level", 100)
        last_seen = device.get("last_seen", datetime.now())
        time_since_last_seen = (datetime.now() - last_seen).total_seconds() / 60  # minutes

        health_score = 100

        # Battery impact
        if battery_level < 20:
            health_score -= 30
        elif battery_level < 50:
            health_score -= 10

        # Connectivity impact
        if time_since_last_seen > 60:
            health_score -= 20

        elif time_since_last_seen > 30 :
            health_score -= 10
        # Firmware impact (simplified)
        firmware = device.get("firmware_version", "1.0.0")
        if firmware.startswith("0."):
            health_score -= 15

        health_score = max(0, health_score)

        return {
            "score": health_score,
            "status": (
                "excellent" if health_score >= 80 else
                "good" if health_score >= 60 else
                "fair" if health_score >= 40 else
                "poor"
            ),
            "factors": await self._identify_health_factors(device, health_score)
        }

    async def _identify_health_factors(self, device: Dict[str, Any], health_score: int) -> List[str]:
        """Identify factors affecting device health"""
        factors = []

        battery = device.get("battery_level", 100)
        if battery < 30:
            factors.append(f"Low battery ({battery}%)")

        last_seen = device.get("last_seen", datetime.now())
        if (datetime.now() - last_seen).total_seconds() > 3600:  # 1 hour
            factors.append("Infrequent communication")

        if health_score < 60:
            factors.append("Needs maintenance attention")

        return factors


    async def _summarize_readings(self, sensor_readings: Dict[str, List[float]]) -> Dict[str, Any]:
        """Summarize sensor readings across devices"""
        summary = {}

        for sensor_type, readings in sensor_readings.items():
            if readings:
                config = self.sensor_types.get(sensor_type, {})
                normal_min, normal_max = config.get("normal_range", (0, 100))

                summary[sensor_type] = {
                    "count": len(readings),
                    "average": statistics.mean(readings),
                    "min": min(readings),
                    "max": max(readings),
                    "in_normal_range": sum(1 for r in readings if normal_min <= r <= normal_max) / len(readings)
                }

        return summary


    async def _generate_historical_reading(self, sensor_type: str, hours_ago: int) -> float:
        """Generate historical sensor reading with daily pattern"""
        base_reading = await self._generate_sensor_reading(sensor_type)

        # Add daily pattern variation
        current_hour = (datetime.now().hour - hours_ago) % 24
        if sensor_type == "temperature":
            # Colder at night, warmer during day
            if 0 <= current_hour <= 6:  # Night
                base_reading -= 3
            elif 12 <= current_hour <= 18:  # Day
                base_reading += 2

        return round(base_reading, 1)


    async def _assess_data_quality(self, historical_readings: List[Dict], current_reading: float) -> str:
        """Assess quality of sensor data"""
        if not historical_readings:
            return "unknown"

        values = [r["value"] for r in historical_readings]

        # Check for missing data
        if len(historical_readings) < 20:
            return "incomplete"

        # Check for outliers
        if current_reading > statistics.mean(values) + 3 * statistics.stdev(values):
            return "anomalous"

        return "good"


    async def _detect_anomalies(self, historical_readings: List[Dict], current_reading: float) -> List[Dict[str, Any]]:
        """Detect anomalies in sensor data"""
        anomalies = []
        values = [r["value"] for r in historical_readings]

        if len(values) < 2:
            return anomalies

        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)

        # Check current reading
        if abs(current_reading - mean_val) > 2 * std_val:
            anomalies.append({
                "type": "current_reading_anomaly",
                "description": f"Current reading {current_reading} differs significantly from historical mean {mean_val:.2f}",
                "severity": "medium"
            })

        # Check for trends in historical data
        if len(values) >= 6:
            recent = values[-6:]
            if all(r > mean_val + std_val for r in recent):
                anomalies.append({
                    "type": "sustained_high_readings",
                    "description": "Sustained high readings detected in recent data",
                    "severity": "low"
                })

        return anomalies


    async def _analyze_trends(self, historical_readings: List[Dict]) -> Dict[str, Any]:
        """Analyze trends in historical sensor data"""
        if len(historical_readings) < 3:
            return {"trend": "insufficient_data"}

        values = [r["value"] for r in historical_readings]

        # Simple trend calculation
        first_half = values[:len(values) // 2]
        second_half = values[len(values) // 2:]

        avg_first = statistics.mean(first_half) if first_half else 0
        avg_second = statistics.mean(second_half) if second_half else 0

        if avg_second > avg_first * 1.1:
            trend = "increasing"
        elif avg_second < avg_first * 0.9:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "volatility": statistics.stdev(values) if len(values) > 1 else 0,
            "seasonal_pattern": await self._detect_seasonal_pattern(historical_readings)
        }


    async def _generate_data_quality_report(self, simulated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data quality report for all sensors"""
        quality_counts = {}
        total_sensors = len(simulated_data)

        for device_data in simulated_data.values():
            quality = device_data.get("data_quality", "unknown")
            quality_counts[quality] = quality_counts.get(quality, 0) + 1

        return {
            "total_sensors": total_sensors,
            "quality_distribution": quality_counts,
            "overall_quality_score": quality_counts.get("good", 0) / total_sensors if total_sensors else 0
        }


    async def _analyze_device_distribution(self, devices: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze distribution of devices by type and location"""
        distribution = {}

        for device in devices:
            device_type = device.get("type", "unknown")
            location = device.get("location", "unknown")

            key = f"{device_type}_{location}"
            distribution[key] = distribution.get(key, 0) + 1

        return distribution


    async def _calculate_average_battery(self, devices: List[Dict[str, Any]]) -> float:
        """Calculate average battery level across devices"""
        battery_levels = [d.get("battery_level", 100) for d in devices]
        return statistics.mean(battery_levels) if battery_levels else 100


    async def _assess_network_coverage(self, devices: List[Dict[str, Any]]) -> str:
        """Assess network coverage quality"""
        online_devices = [d for d in devices if d.get("status") == "online"]
        online_ratio = len(online_devices) / len(devices) if devices else 0

        if online_ratio >= 0.9:
            return "excellent"
        elif online_ratio >= 0.7:
            return "good"
        elif online_ratio >= 0.5:
            return "fair"
        else:
            return "poor"


    async def _estimate_data_throughput(self, devices: List[Dict[str, Any]]) -> float:
        """Estimate data throughput in kbps"""
        sensor_devices = [d for d in devices if d.get("type") == "sensor"]
        # Rough estimate: 1 kbps per sensor device
        return len(sensor_devices) * 1.0


    async def _estimate_network_latency(self, devices: List[Dict[str, Any]]) -> float:
        """Estimate network latency in milliseconds"""
        gateway_devices = [d for d in devices if d.get("type") == "gateway"]
        if not gateway_devices:
            return 500.0  # High latency without gateways

        # Base latency + additional latency per device
        base_latency = 50.0
        device_latency = len(devices) * 0.5
        return base_latency + device_latency


    async def _assess_network_security(self, devices: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess network security status"""
        outdated_firmware = 0
        weak_battery = 0

        for device in devices:
            firmware = device.get("firmware_version", "1.0.0")
            if firmware.startswith("0.") or "beta" in firmware.lower():
                outdated_firmware += 1

            if device.get("battery_level", 100) < 20:
                weak_battery += 1

        return {
            "security_score": max(0, 100 - outdated_firmware * 10 - weak_battery * 5),
            "outdated_devices": outdated_firmware,
            "devices_needing_attention": outdated_firmware + weak_battery,
            "recommendations": [
                "Update firmware on outdated devices",
                "Replace batteries on low-power devices"
            ]
        }


    async def _identify_optimization_opportunities(self, devices: List[Dict[str, Any]]) -> List[str]:
        """Identify opportunities for network optimization"""
        opportunities = []

        gateway_count = len([d for d in devices if d.get("type") == "gateway"])
        if gateway_count == 0:
            opportunities.append("Add gateway device to improve network reliability")

        low_battery_count = len([d for d in devices if d.get("battery_level", 100) < 30])
        if low_battery_count > len(devices) * 0.3:
            opportunities.append("Schedule battery replacement for multiple devices")

        offline_count = len([d for d in devices if d.get("status") == "offline"])
        if offline_count > len(devices) * 0.2:
            opportunities.append("Investigate connectivity issues with offline devices")

        return opportunities


    async def _identify_critical_issues(self, devices: List[Dict[str, Any]], device_health: Dict[str, Any]) -> List[str]:
        """Identify critical issues requiring immediate attention"""
        critical_issues = []

        for device in devices:
            health = device_health.get(device["id"], {})
            if health.get("score", 100) < 30:
                critical_issues.append(f"Critical health issue with {device['id']} (score: {health.get('score', 0)})")

            if device.get("battery_level", 100) < 10:
                critical_issues.append(f"Very low battery on {device['id']} ({device.get('battery_level', 0)}%)")

        return critical_issues[:3]  # Return top 3 critical issues


    async def _generate_maintenance_recommendations(self, devices: List[Dict[str, Any]], device_health: Dict[str, Any]) -> List[str]:
        """Generate maintenance recommendations"""
        recommendations = []

        low_battery_devices = [d for d in devices if d.get("battery_level", 100) < 40]
        if low_battery_devices:
            recommendations.append(f"Schedule battery replacement for {len(low_battery_devices)} devices")

        poor_health_devices = [d for d in devices if device_health.get(d["id"], {}).get("score", 100) < 50]
        if poor_health_devices:
            recommendations.append(f"Perform maintenance on {len(poor_health_devices)} devices with poor health")

        return recommendations


    async def _analyze_battery_status(self, devices: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall battery status"""
        battery_levels = [d.get("battery_level", 100) for d in devices]

        return {
            "average_battery": statistics.mean(battery_levels) if battery_levels else 100,
            "min_battery": min(battery_levels) if battery_levels else 100,
            "devices_below_20%": len([b for b in battery_levels if b < 20]),
            "battery_health": "good" if statistics.mean(battery_levels) > 60 else "fair" if statistics.mean(
                battery_levels) > 30 else "poor"
        }


    async def _calculate_network_load(self, devices: List[Dict[str, Any]]) -> str:
        """Calculate current network load"""
        online_devices = len([d for d in devices if d.get("status") == "online"])
        total_devices = len(devices)

        load_ratio = online_devices / total_devices if total_devices else 0

        if load_ratio > 0.9:
            return "high"
        elif load_ratio > 0.6:
            return "medium"
        else:
            return "low"


    async def _suggest_network_improvements(self, devices: List[Dict[str, Any]]) -> List[str]:
        """Suggest network improvements"""
        suggestions = []

        gateway_count = len([d for d in devices if d.get("type") == "gateway"])
        if gateway_count == 0:
            suggestions.append("Add at least one gateway device")
        elif gateway_count < len(devices) / 10:
            suggestions.append("Consider adding more gateway devices for better coverage")

        offline_count = len([d for d in devices if d.get("status") == "offline"])
        if offline_count > len(devices) * 0.1:
            suggestions.append("Investigate why multiple devices are offline")

        return suggestions


    async def _detect_seasonal_pattern(self, historical_readings: List[Dict]) -> str:
        """Detect seasonal patterns in sensor data"""
        if len(historical_readings) < 24:
            return "insufficient_data"

        # Simple pattern detection based on time of day
        day_readings = [r for r in historical_readings if 6 <= datetime.fromisoformat(r["timestamp"]).hour <= 18]
        night_readings = [r for r in historical_readings if
                          datetime.fromisoformat(r["timestamp"]).hour < 6 or datetime.fromisoformat(
                              r["timestamp"]).hour > 18]

        if day_readings and night_readings:
            day_avg = statistics.mean([r["value"] for r in day_readings])
            night_avg = statistics.mean([r["value"] for r in night_readings])

            # Consider difference significant if >10%
            if abs(day_avg - night_avg) > abs(day_avg) * 0.1:
                return "diurnal"

        return "none"


    async def _generate_iot_recommendations(self, devices: List[Dict[str, Any]], health_report: Dict[str, Any]) -> List[str]:
        """Generate IoT-specific recommendations"""
        recommendations = []

        overall_health = health_report.get("overall_health", "unknown")
        if overall_health in ["fair", "poor"]:
            recommendations.append("Prioritize device maintenance and health improvement")

        battery_status = health_report.get("battery_status", {})
        if battery_status.get("devices_below_20%", 0) > 0:
            recommendations.append(f"Replace batteries on {battery_status['devices_below_20%']} devices")

        network_status = await self._assess_network_status(devices)
        if network_status.get("network_status") == "degraded":
            recommendations.append("Investigate and resolve network connectivity issues")

        if not recommendations:
            recommendations.append("IoT network operating normally. Continue regular monitoring.")

        return recommendations
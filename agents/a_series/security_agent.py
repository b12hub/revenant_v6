# # /agents/a_series/security_agent.py
# import re
# from asyncio import sleep, get_event_loop
# from typing import Dict, Any
# import asyncio
# from core.agent_base import RevenantAgentBase
#
#
# class SecurityAgent(RevenantAgentBase):
#     def __init__(self):
#         super().__init__(
#             name="SecurityAgent",
#             description="Analyzes security threats, validates inputs, and protects the Revenant ecosystem from malicious activity."
#         )
#         self.threat_patterns = []
#         self.rate_limits = {}
#
#     async def setup(self):
#         # Initialize security databases and patterns
#         self.threat_patterns = [
#             r"(?i)(password|token|key)\s*[=:]\s*['\"][^'\"]+['\"]",
#             r"(?i)(select|insert|update|delete|drop|union).*where",
#             r"(?i)(<script|javascript:)|(onload|onerror)=",
#             r"(?i)(\.\./|\.\.\\|~/|/etc/passwd)",
#         ]
#         self.rate_limits = {}
#         await sleep(0.1)
#
#     async def run(self, input_data: dict):
#         try:
#             # Input validation
#             if not self._validate_input_structure(input_data):
#                 raise ValueError("Invalid input structure")
#
#             # Threat analysis
#             threat_analysis = await self._analyze_threats(input_data)
#
#             # Rate limiting check
#             rate_check = await self._check_rate_limits(input_data)
#
#             result = {
#                 "threat_level": threat_analysis["threat_level"],
#                 "threats_detected": threat_analysis["threats_detected"],
#                 "rate_limit_status": rate_check["status"],
#                 "recommendations": threat_analysis["recommendations"],
#                 "input_hash": self._hash_input(input_data)
#             }
#
#             return {
#                 "agent": self.name,
#                 "status": "ok",
#                 "summary": f"Security analysis completed - Threat level: {threat_analysis['threat_level']}",
#                 "data": result
#             }
#
#         except Exception as e:
#             return await self.on_error(e)
#
#     async def _analyze_threats(self, input_data: dict) -> Dict[str, Any]:
#         threats = []
#         input_str = str(input_data)
#
#         for pattern in self.threat_patterns:
#             if re.search(pattern, input_str):
#                 threats.append(f"Pattern matched: {pattern}")
#
#         threat_level = "high" if len(threats) > 2 else "medium" if threats else "low"
#
#         return {
#             "threat_level": threat_level,
#             "threats_detected": threats,
#             "recommendations": ["Sanitize input", "Validate user session"] if threats else ["Input appears safe"]
#         }
#
#     async def _check_rate_limits(self, input_data: dict) -> Dict[str, Any]:
#         user_id = input_data.get("sessionId", "anonymous")
#         current_time = get_event_loop().time()
#
#         if user_id not in self.rate_limits:
#             self.rate_limits[user_id] = []
#
#         # Remove old entries (last minute)
#         self.rate_limits[user_id] = [
#             t for t in self.rate_limits[user_id]
#             if current_time - t < 60
#         ]
#
#         if len(self.rate_limits[user_id]) >= 10:  # 10 requests per minute
#             return {"status": "blocked", "message": "Rate limit exceeded"}
#
#         self.rate_limits[user_id].append(current_time)
#         return {"status": "allowed", "count": len(self.rate_limits[user_id])}
#
#     def _validate_input_structure(self, input_data: dict) -> bool:
#         required_fields = ["userInput", "sessionId"]
#         return all(field in input_data for field in required_fields)
#
#     def _hash_input(self, input_data: dict) -> str:
#         return str(hash(str(input_data)))


# /agents/a_series/security_agent.py
import re
from asyncio import sleep, get_event_loop
from typing import Dict, Any
from core.agent_base import RevenantAgentBase


class SecurityAgent(RevenantAgentBase):
    """
    Analyzes security threats, validates inputs, and protects the Revenant ecosystem.

    Input:
        - userInput (str): User-provided input to analyze
        - sessionId (str): Session identifier for rate limiting

    Output:
        - threat_level (str): "low", "medium", or "high"
        - threats_detected (list): List of detected threat patterns
        - rate_limit_status (dict): Rate limiting information
        - recommendations (list): Security recommendations
    """

    metadata = {
        "name": "SecurityAgent",
        "series": "a_series",
        "version": "0.1.0",
        "description": "Analyzes security threats, validates inputs, and protects the Revenant ecosystem from malicious activity."
    }

    def __init__(self):
        super().__init__(
            name="SecurityAgent",
            description="Analyzes security threats, validates inputs, and protects the Revenant ecosystem from malicious activity."
        )
        self.threat_patterns = []
        self.rate_limits = {}
        self.metadata = SecurityAgent.metadata

    async def setup(self):
        # Initialize security databases and patterns
        self.threat_patterns = [
            r"(?i)(password|token|key)\s*[=:]\s*['\"][^'\"]+['\"]",
            r"(?i)(select|insert|update|delete|drop|union).*where",
            r"(?i)(<script|javascript:)|(onload|onerror)=",
            r"(?i)(\.\./|\.\.\\|~/|/etc/passwd)",
        ]
        self.rate_limits = {}
        await sleep(0.1)

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Input validation
            if not self._validate_input_structure(input_data):
                raise ValueError("Invalid input structure")

            # Threat analysis
            threat_analysis = await self._analyze_threats(input_data)

            # Rate limiting check
            rate_check = await self._check_rate_limits(input_data)

            result = {
                "threat_level": threat_analysis["threat_level"],
                "threats_detected": threat_analysis["threats_detected"],
                "rate_limit_status": rate_check["status"],
                "recommendations": threat_analysis["recommendations"],
                "input_hash": self._hash_input(input_data)
            }

            return {
                "agent": self.name,
                "status": "ok",
                "summary": f"Security analysis completed - Threat level: {threat_analysis['threat_level']}",
                "data": result
            }

        except Exception as e:
            return await self.on_error(e)

    async def _analyze_threats(self, input_data: dict) -> Dict[str, Any]:
        threats = []
        input_str = str(input_data)

        for pattern in self.threat_patterns:
            if re.search(pattern, input_str):
                threats.append(f"Pattern matched: {pattern}")

        threat_level = "high" if len(threats) > 2 else "medium" if threats else "low"

        return {
            "threat_level": threat_level,
            "threats_detected": threats,
            "recommendations": ["Sanitize input", "Validate user session"] if threats else ["Input appears safe"]
        }

    async def _check_rate_limits(self, input_data: dict) -> Dict[str, Any]:
        user_id = input_data.get("sessionId", "anonymous")
        current_time = get_event_loop().time()

        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = []

        # Remove old entries (last minute)
        self.rate_limits[user_id] = [
            t for t in self.rate_limits[user_id]
            if current_time - t < 60
        ]

        if len(self.rate_limits[user_id]) >= 10:  # 10 requests per minute
            return {"status": "blocked", "message": "Rate limit exceeded"}

        self.rate_limits[user_id].append(current_time)
        return {"status": "allowed", "count": len(self.rate_limits[user_id])}

    def _validate_input_structure(self, input_data: dict) -> bool:
        required_fields = ["userInput", "sessionId"]
        return all(field in input_data for field in required_fields)

    def _hash_input(self, input_data: dict) -> str:
        return str(hash(str(input_data)))
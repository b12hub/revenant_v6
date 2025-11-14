# agents/d_series/governance_agent.py
import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading

from core.agent_base import RevenantAgentBase


class PolicyDecision(Enum):
    ALLOW = "allow"
    DENY = "deny"


class RemediationAction(Enum):
    RETRY = "retry"
    PAUSE = "pause"
    ESCALATE = "escalate"


@dataclass
class PolicyRule:
    id: str
    description: str
    conditions: List[Dict[str, Any]]
    action: str  # "allow" or "deny"
    priority: int = 1


@dataclass
class AuditEntry:
    timestamp: float
    trace_id: str
    actor: str
    action: str
    resource: str
    decision: str
    reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RateLimitConfig:
    key: str
    limit: int
    window: int  # seconds
    current: int = 0
    window_start: float = 0


class GovernanceAgent(RevenantAgentBase):
    metadata = {
        "name": "GovernanceAgent",
        "series": "d_series",
        "version": "0.1.0",
        "description": "Policy, RBAC, rate-limits, audit and remediation agent for D-Series orchestration",
        "module": "agents.d_series.governance_agent"
    }

    def __init__(self, storage=None, logger: Optional[logging.Logger] = None):
        super().__init__(
            name=self.metadata['name'] ,
            description= self.metadata['description']
        )
        self.storage = storage or {}
        self.logger = logger or logging.getLogger(__name__)

        # Policy engine state
        self.policies: Dict[str, PolicyRule] = {}
        self._policy_lock = threading.Lock()

        # Rate limiting state
        self.rate_limits: Dict[str, RateLimitConfig] = {}
        self._rate_limit_lock = threading.Lock()

        # Audit log configuration
        self.audit_log_path = Path("logs/governance_audit.log")
        self.audit_log_path.parent.mkdir(exist_ok=True)

        # RBAC roles (simple in-memory storage)
        self.roles: Dict[str, List[str]] = {}  # role -> list of permissions
        self.user_roles: Dict[str, List[str]] = {}  # user -> list of roles

        # Initialize with default policies
        self._initialize_default_policies()

    def _initialize_default_policies(self):
        """Initialize with some default security policies"""
        default_policies = [
            PolicyRule(
                id="block-untrusted-external",
                description="Block external call nodes flagged untrusted",
                conditions=[
                    {"field": "node.type", "op": "==", "value": "external_http"},
                    {"field": "node.trusted", "op": "==", "value": False}
                ],
                action="deny",
                priority=10
            ),
            PolicyRule(
                id="require-trace-id",
                description="Require trace_id for all agent calls",
                conditions=[
                    {"field": "context.trace_id", "op": "exists", "value": True},
                    {"field": "context.trace_id", "op": "!=", "value": ""}
                ],
                action="allow",
                priority=5
            )
        ]

        for policy in default_policies:
            self.policies[policy.id] = policy

    async def run(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute governance commands.

        Supported commands:
        - check_policy: Evaluate policies against context
        - enforce_rate_limit: Check and enforce rate limits
        - audit_event: Record audit event
        - remediate: Execute remediation action

        Args:
            command: Governance command dictionary

        Returns:
            Dictionary with command result
        """
        action = command.get("action")
        trace_id = command.get("trace_id", "unknown")

        self.logger.info(
            f"Executing governance action: {action}",
            extra={"trace_id": trace_id, "governance_action": action}
        )

        try:
            if action == "check_policy":
                return await self._check_policy(command, trace_id)
            elif action == "enforce_rate_limit":
                return await self._enforce_rate_limit(command, trace_id)
            elif action == "audit_event":
                return await self._audit_event(command, trace_id)
            elif action == "remediate":
                return await self._remediate(command, trace_id)
            else:
                return {
                    "success": False,
                    "error": f"Unknown governance action: {action}",
                    "trace_id": trace_id
                }
        except Exception as e:
            self.logger.error(
                f"Governance action failed: {e}",
                extra={"trace_id": trace_id, "governance_action": action}
            )
            return {
                "success": False,
                "error": str(e),
                "trace_id": trace_id
            }

    async def _check_policy(self, command: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
        """Evaluate policies against context"""
        policy_id = command.get("policy_id")
        context = command.get("context", {})

        if policy_id:
            # Check specific policy
            policy = self.policies.get(policy_id)
            if not policy:
                return {
                    "success": False,
                    "error": f"Policy not found: {policy_id}",
                    "trace_id": trace_id
                }

            decision, reason = self._evaluate_policy(policy, context)
        else:
            # Evaluate all applicable policies
            decision, reason = self._evaluate_all_policies(context)

        # Record audit event for policy checks
        await self._audit_event({
            "event": {
                "actor": context.get("actor", "system"),
                "action": "policy_check",
                "resource": context.get("resource", "unknown"),
                "decision": decision.value,
                "reason": reason,
                "policy_id": policy_id,
                "context": context
            }
        }, trace_id)

        return {
            "success": True,
            "decision": decision.value,
            "reason": reason,
            "policy_id": policy_id,
            "trace_id": trace_id
        }

    def _evaluate_policy(self, policy: PolicyRule, context: Dict[str, Any]) -> Tuple[PolicyDecision, str]:
        """Evaluate a single policy rule against context"""
        try:
            conditions_met = all(self._evaluate_condition(condition, context)
                                 for condition in policy.conditions)

            if conditions_met:
                decision = PolicyDecision.ALLOW if policy.action == "allow" else PolicyDecision.DENY
                return decision, f"Policy {policy.id} conditions met"
            else:
                return PolicyDecision.ALLOW, f"Policy {policy.id} conditions not met"

        except Exception as e:
            return PolicyDecision.DENY, f"Policy evaluation error: {str(e)}"

    def _evaluate_all_policies(self, context: Dict[str, Any]) -> Tuple[PolicyDecision, str]:
        """Evaluate all policies by priority (deny takes precedence)"""
        with self._policy_lock:
            sorted_policies = sorted(self.policies.values(),
                                     key=lambda p: p.priority, reverse=True)

        for policy in sorted_policies:
            conditions_met = all(self._evaluate_condition(condition, context)
                                 for condition in policy.conditions)

            if conditions_met:
                decision = PolicyDecision.ALLOW if policy.action == "allow" else PolicyDecision.DENY
                return decision, f"Policy {policy.id} triggered"

        return PolicyDecision.ALLOW, "No policies matched, default allow"

    def _evaluate_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate a single condition against context"""
        field = condition["field"]
        op = condition["op"]
        expected_value = condition.get("value")

        # Navigate nested fields using dot notation
        current_value = context
        for key in field.split('.'):
            if isinstance(current_value, dict) and key in current_value:
                current_value = current_value[key]
            else:
                current_value = None
                break

        # Handle different operators
        if op == "==":
            return current_value == expected_value
        elif op == "!=":
            return current_value != expected_value
        elif op == ">":
            return current_value > expected_value
        elif op == "<":
            return current_value < expected_value
        elif op == ">=":
            return current_value >= expected_value
        elif op == "<=":
            return current_value <= expected_value
        elif op == "in":
            return current_value in expected_value
        elif op == "not in":
            return current_value not in expected_value
        elif op == "exists":
            return expected_value is True and current_value is not None
        elif op == "regex":
            import re
            return bool(re.match(expected_value, str(current_value)))
        else:
            raise ValueError(f"Unsupported operator: {op}")

    async def _enforce_rate_limit(self, command: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
        """Enforce rate limiting using token bucket algorithm"""
        key = command["key"]
        limit = command["limit"]
        window = command.get("window", 60)  # default 60 seconds

        with self._rate_limit_lock:
            current_time = time.time()

            if key not in self.rate_limits:
                # Initialize new rate limit
                self.rate_limits[key] = RateLimitConfig(
                    key=key, limit=limit, window=window,
                    current=limit, window_start=current_time
                )

            config = self.rate_limits[key]

            # Check if window has expired
            if current_time - config.window_start >= window:
                # Reset token bucket
                config.current = limit
                config.window_start = current_time

            if config.current <= 0:
                # Rate limit exceeded
                await self._audit_event({
                    "event": {
                        "actor": "system",
                        "action": "rate_limit_exceeded",
                        "resource": key,
                        "decision": "deny",
                        "reason": f"Rate limit exceeded: {limit} per {window}s"
                    }
                }, trace_id)

                return {
                    "success": False,
                    "allowed": False,
                    "remaining": 0,
                    "reset_in": window - (current_time - config.window_start),
                    "trace_id": trace_id
                }

            # Consume one token
            config.current -= 1

            return {
                "success": True,
                "allowed": True,
                "remaining": config.current,
                "reset_in": window - (current_time - config.window_start),
                "trace_id": trace_id
            }

    async def _audit_event(self, command: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
        """Record audit event to log file and structured logger"""
        event_data = command["event"]

        audit_entry = AuditEntry(
            timestamp=time.time(),
            trace_id=trace_id,
            actor=event_data.get("actor", "unknown"),
            action=event_data.get("action", "unknown"),
            resource=event_data.get("resource", "unknown"),
            decision=event_data.get("decision", "unknown"),
            reason=event_data.get("reason"),
            metadata=event_data.get("metadata")
        )

        # Write to audit log file
        try:
            with open(self.audit_log_path, "a") as audit_file:
                audit_file.write(json.dumps(asdict(audit_entry)) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write audit log: {e}")

        # Log to structured logger
        self.logger.info(
            "Audit event recorded",
            extra={
                "trace_id": trace_id,
                "audit_actor": audit_entry.actor,
                "audit_action": audit_entry.action,
                "audit_resource": audit_entry.resource,
                "audit_decision": audit_entry.decision,
                "audit_reason": audit_entry.reason
            }
        )

        return {
            "success": True,
            "audit_id": f"audit_{int(audit_entry.timestamp)}_{trace_id}",
            "trace_id": trace_id
        }

    async def _remediate(self, command: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
        """Execute remediation action"""
        issue = command["issue"]
        action = issue.get("remediation_action", "retry")

        remediation_action = RemediationAction(action)

        if remediation_action == RemediationAction.RETRY:
            result = await self._remediate_retry(issue, trace_id)
        elif remediation_action == RemediationAction.PAUSE:
            result = await self._remediate_pause(issue, trace_id)
        elif remediation_action == RemediationAction.ESCALATE:
            result = await self._remediate_escalate(issue, trace_id)
        else:
            result = {"success": False, "error": f"Unknown remediation action: {action}"}

        # Audit the remediation action
        await self._audit_event({
            "event": {
                "actor": "governance_agent",
                "action": "remediation",
                "resource": issue.get("resource", "unknown"),
                "decision": "execute",
                "reason": f"Remediation: {action} for issue: {issue.get('description')}",
                "metadata": {"remediation_result": result}
            }
        }, trace_id)

        return {
            "success": True,
            "remediation_action": action,
            "remediation_result": result,
            "trace_id": trace_id
        }

    async def _remediate_retry(self, issue: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
        """Retry remediation - simulate retry logic"""
        max_retries = issue.get("max_retries", 3)
        retry_count = issue.get("retry_count", 0)

        if retry_count < max_retries:
            return {
                "action": "retry",
                "retry_count": retry_count + 1,
                "message": f"Scheduling retry {retry_count + 1} of {max_retries}"
            }
        else:
            return {
                "action": "escalate",
                "reason": f"Max retries ({max_retries}) exceeded"
            }

    async def _remediate_pause(self, issue: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
        """Pause remediation - simulate circuit breaker pattern"""
        resource = issue.get("resource", "unknown")
        pause_duration = issue.get("pause_duration", 300)  # 5 minutes default

        return {
            "action": "pause",
            "resource": resource,
            "pause_duration": pause_duration,
            "message": f"Resource {resource} paused for {pause_duration}s"
        }

    async def _remediate_escalate(self, issue: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
        """Escalate remediation - simulate alert escalation"""
        severity = issue.get("severity", "medium")
        channel = issue.get("channel", "ops_team")

        return {
            "action": "escalate",
            "severity": severity,
            "channel": channel,
            "message": f"Issue escalated to {channel} with severity {severity}"
        }

    # Admin API methods
    def add_policy(self, policy_definition: Dict[str, Any]) -> Dict[str, Any]:
        """Add a new policy rule"""
        try:
            policy = PolicyRule(**policy_definition)

            with self._policy_lock:
                self.policies[policy.id] = policy

            self.logger.info(f"Policy added: {policy.id}")
            return {"success": True, "policy_id": policy.id}

        except Exception as e:
            self.logger.error(f"Failed to add policy: {e}")
            return {"success": False, "error": str(e)}

    def remove_policy(self, policy_id: str) -> Dict[str, Any]:
        """Remove a policy rule"""
        with self._policy_lock:
            if policy_id in self.policies:
                del self.policies[policy_id]
                self.logger.info(f"Policy removed: {policy_id}")
                return {"success": True, "policy_id": policy_id}
            else:
                return {"success": False, "error": f"Policy not found: {policy_id}"}

    def list_policies(self) -> Dict[str, Any]:
        """List all policy rules"""
        with self._policy_lock:
            policies = [asdict(policy) for policy in self.policies.values()]

        return {"success": True, "policies": policies}

    def check_rbac(self, subject: str, resource: str, verb: str) -> Dict[str, Any]:
        """Check RBAC permissions"""
        # Simple RBAC implementation
        user_roles = self.user_roles.get(subject, [])
        user_permissions = set()

        for role in user_roles:
            user_permissions.update(self.roles.get(role, []))

        required_permission = f"{resource}:{verb}"
        allowed = required_permission in user_permissions

        return {
            "success": True,
            "allowed": allowed,
            "subject": subject,
            "resource": resource,
            "verb": verb,
            "roles": user_roles,
            "permissions": list(user_permissions)
        }

    def audit(self, query: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query audit logs (simple implementation)"""
        try:
            # This is a simplified implementation - in production, use proper log querying
            if not self.audit_log_path.exists():
                return {"success": True, "audit_entries": []}

            entries = []
            with open(self.audit_log_path, "r") as audit_file:
                for line in audit_file:
                    if line.strip():
                        entries.append(json.loads(line.strip()))

            # Simple filtering (in production, use proper query engine)
            if query:
                filtered_entries = []
                for entry in entries:
                    match = True
                    for key, value in query.items():
                        if entry.get(key) != value:
                            match = False
                            break
                    if match:
                        filtered_entries.append(entry)
                entries = filtered_entries

            return {"success": True, "audit_entries": entries}

        except Exception as e:
            return {"success": False, "error": str(e)}
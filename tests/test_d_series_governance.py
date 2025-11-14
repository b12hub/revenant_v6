# tests/test_d_series_governance.py
import pytest
import asyncio


from agents.d_series.governance_agent import (
    GovernanceAgent,
    PolicyRule,
    PolicyDecision
)


@pytest.fixture
def governance_agent():
    """Create governance agent instance for testing"""
    agent = GovernanceAgent()
    # Clear default policies for clean tests
    agent.policies.clear()
    return agent


@pytest.fixture
def sample_policy():
    """Sample policy for testing"""
    return {
        "id": "test-policy",
        "description": "Test policy for unit tests",
        "conditions": [
            {"field": "user.role", "op": "==", "value": "admin"},
            {"field": "resource.sensitivity", "op": "==", "value": "high"}
        ],
        "action": "allow",
        "priority": 5
    }


@pytest.mark.asyncio
async def test_policy_creation_and_enforcement(governance_agent, sample_policy):
    """Test policy creation and enforcement"""
    # Add policy
    result = governance_agent.add_policy(sample_policy)
    assert result["success"] is True

    # Test policy evaluation - conditions met
    check_result = await governance_agent.run({
        "action": "check_policy",
        "policy_id": "test-policy",
        "context": {
            "user": {"role": "admin"},
            "resource": {"sensitivity": "high"}
        },
        "trace_id": "test_trace_1"
    })

    assert check_result["success"] is True
    assert check_result["decision"] == "allow"

    # Test policy evaluation - conditions not met
    check_result = await governance_agent.run({
        "action": "check_policy",
        "policy_id": "test-policy",
        "context": {
            "user": {"role": "user"},
            "resource": {"sensitivity": "high"}
        },
        "trace_id": "test_trace_2"
    })

    assert check_result["success"] is True
    assert check_result["decision"] == "allow"  # Default allow when conditions not met


@pytest.mark.asyncio
async def test_rbac_checks(governance_agent):
    """Test RBAC permission checks"""
    # Setup roles and permissions
    governance_agent.roles = {
        "admin": ["users:read", "users:write", "system:admin"],
        "user": ["users:read", "profile:write"]
    }
    governance_agent.user_roles = {
        "alice": ["admin", "user"],
        "bob": ["user"]
    }

    # Test admin permissions
    result = governance_agent.check_rbac("alice", "system", "admin")
    assert result["allowed"] is True
    assert "admin" in result["roles"]

    # Test user permissions
    result = governance_agent.check_rbac("bob", "users", "read")
    assert result["allowed"] is True

    # Test denied permission
    result = governance_agent.check_rbac("bob", "system", "admin")
    assert result["allowed"] is False


@pytest.mark.asyncio
async def test_rate_limiting_logic(governance_agent):
    """Test rate limiting functionality"""
    rate_limit_key = "api_user_123"

    # First request should be allowed
    result = await governance_agent.run({
        "action": "enforce_rate_limit",
        "key": rate_limit_key,
        "limit": 5,
        "window": 60,
        "trace_id": "test_rate_1"
    })

    assert result["success"] is True
    assert result["allowed"] is True
    assert result["remaining"] == 4

    # Exhaust the rate limit
    for i in range(4):
        result = await governance_agent.run({
            "action": "enforce_rate_limit",
            "key": rate_limit_key,
            "limit": 5,
            "window": 60,
            "trace_id": f"test_rate_{i + 2}"
        })

    # Next request should be denied
    result = await governance_agent.run({
        "action": "enforce_rate_limit",
        "key": rate_limit_key,
        "limit": 5,
        "window": 60,
        "trace_id": "test_rate_denied"
    })

    assert result["success"] is False
    assert result["allowed"] is False
    assert result["remaining"] == 0


@pytest.mark.asyncio
async def test_audit_logs_generation(governance_agent):
    """Test audit log generation"""
    audit_event = {
        "actor": "test_user",
        "action": "data_access",
        "resource": "sensitive_data",
        "decision": "allow",
        "reason": "Authorized access",
        "metadata": {"ip": "127.0.0.1"}
    }

    result = await governance_agent.run({
        "action": "audit_event",
        "event": audit_event,
        "trace_id": "test_audit_1"
    })

    assert result["success"] is True
    assert "audit_id" in result

    # Verify audit log file was created
    assert governance_agent.audit_log_path.exists()

    # Test audit query
    query_result = governance_agent.audit()
    assert query_result["success"] is True
    assert len(query_result["audit_entries"]) >= 1


@pytest.mark.asyncio
async def test_remediation_primitive_simulation(governance_agent):
    """Test remediation actions"""
    # Test retry remediation
    retry_issue = {
        "remediation_action": "retry",
        "resource": "failing_service",
        "description": "Service timeout",
        "max_retries": 3,
        "retry_count": 1
    }

    result = await governance_agent.run({
        "action": "remediate",
        "issue": retry_issue,
        "trace_id": "test_remediate_1"
    })

    assert result["success"] is True
    assert result["remediation_action"] == "retry"
    assert "retry_count" in result["remediation_result"]

    # Test pause remediation
    pause_issue = {
        "remediation_action": "pause",
        "resource": "circuit_breaker",
        "description": "Too many failures",
        "pause_duration": 300
    }

    result = await governance_agent.run({
        "action": "remediate",
        "issue": pause_issue,
        "trace_id": "test_remediate_2"
    })

    assert result["success"] is True
    assert result["remediation_action"] == "pause"
    assert "pause_duration" in result["remediation_result"]

    # Test escalate remediation
    escalate_issue = {
        "remediation_action": "escalate",
        "resource": "critical_system",
        "description": "Security breach detected",
        "severity": "high",
        "channel": "security_team"
    }

    result = await governance_agent.run({
        "action": "remediate",
        "issue": escalate_issue,
        "trace_id": "test_remediate_3"
    })

    assert result["success"] is True
    assert result["remediation_action"] == "escalate"
    assert "severity" in result["remediation_result"]


@pytest.mark.asyncio
async def test_policy_condition_operators(governance_agent):
    """Test various policy condition operators"""
    # Test equality operator
    policy = PolicyRule(
        id="equality-test",
        description="Test equality operator",
        conditions=[{"field": "value", "op": "==", "value": "expected"}],
        action="allow"
    )
    governance_agent.add_policy(policy.__dict__)

    result = await governance_agent.run({
        "action": "check_policy",
        "policy_id": "equality-test",
        "context": {"value": "expected"},
        "trace_id": "test_operators_1"
    })
    assert result["decision"] == "allow"

    # Test inequality operator
    policy = PolicyRule(
        id="inequality-test",
        description="Test inequality operator",
        conditions=[{"field": "value", "op": "!=", "value": "unexpected"}],
        action="allow"
    )
    governance_agent.add_policy(policy.__dict__)

    result = await governance_agent.run({
        "action": "check_policy",
        "policy_id": "inequality-test",
        "context": {"value": "expected"},
        "trace_id": "test_operators_2"
    })
    assert result["decision"] == "allow"

    # Test exists operator
    policy = PolicyRule(
        id="exists-test",
        description="Test exists operator",
        conditions=[{"field": "required_field", "op": "exists", "value": True}],
        action="allow"
    )
    governance_agent.add_policy(policy.__dict__)

    result = await governance_agent.run({
        "action": "check_policy",
        "policy_id": "exists-test",
        "context": {"required_field": "present"},
        "trace_id": "test_operators_3"
    })
    assert result["decision"] == "allow"


@pytest.mark.asyncio
async def test_complex_policy_evaluation(governance_agent):
    """Test evaluation of multiple policies with priorities"""
    # Add deny policy with high priority
    governance_agent.add_policy({
        "id": "high-priority-deny",
        "description": "High priority deny rule",
        "conditions": [{"field": "risk_level", "op": "==", "value": "high"}],
        "action": "deny",
        "priority": 100
    })

    # Add allow policy with lower priority
    governance_agent.add_policy({
        "id": "low-priority-allow",
        "description": "Low priority allow rule",
        "conditions": [{"field": "user.trusted", "op": "==", "value": True}],
        "action": "allow",
        "priority": 1
    })

    # Test that deny takes precedence
    result = await governance_agent.run({
        "action": "check_policy",
        "context": {
            "risk_level": "high",
            "user": {"trusted": True}
        },
        "trace_id": "test_priority_1"
    })

    assert result["decision"] == "deny"
    assert "high-priority-deny" in result["reason"]


@pytest.mark.asyncio
async def test_policy_management_api(governance_agent, sample_policy):
    """Test policy management admin API"""
    # Test add policy
    result = governance_agent.add_policy(sample_policy)
    assert result["success"] is True

    # Test list policies
    result = governance_agent.list_policies()
    assert result["success"] is True
    assert len(result["policies"]) == 1
    assert result["policies"][0]["id"] == "test-policy"

    # Test remove policy
    result = governance_agent.remove_policy("test-policy")
    assert result["success"] is True

    # Test remove non-existent policy
    result = governance_agent.remove_policy("non-existent")
    assert result["success"] is False


def test_nested_field_access(governance_agent):
    """Test accessing nested fields in policy conditions"""
    policy = PolicyRule(
        id="nested-test",
        description="Test nested field access",
        conditions=[{"field": "user.profile.role", "op": "==", "value": "admin"}],
        action="allow"
    )

    governance_agent.add_policy(policy.__dict__)

    # This would be tested in an async test, but we test the condition evaluation directly
    context = {
        "user": {
            "profile": {
                "role": "admin"
            }
        }
    }

    decision, reason = governance_agent._evaluate_policy(policy, context)
    assert decision == PolicyDecision.ALLOW


@pytest.mark.asyncio
async def test_rate_limit_window_reset(governance_agent):
    """Test rate limit window reset behavior"""
    rate_limit_key = "reset_test"

    # Set up rate limit with very short window for testing
    await governance_agent.run({
        "action": "enforce_rate_limit",
        "key": rate_limit_key,
        "limit": 2,
        "window": 1,  # 1 second window
        "trace_id": "test_reset_1"
    })

    # Wait for window to reset
    await asyncio.sleep(1.1)

    # Should be allowed again after window reset
    result = await governance_agent.run({
        "action": "enforce_rate_limit",
        "key": rate_limit_key,
        "limit": 2,
        "window": 1,
        "trace_id": "test_reset_2"
    })

    assert result["allowed"] is True
    assert result["remaining"] == 1
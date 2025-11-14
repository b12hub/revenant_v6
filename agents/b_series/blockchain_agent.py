# /agents/b_series/blockchain_agent.py
from core.agent_base import RevenantAgentBase
import asyncio
from typing import Dict, List, Any
from datetime import datetime, timedelta
import hashlib
import random
import statistics


class BlockchainAgent(RevenantAgentBase):
    """Wallet lookup, contract data analysis, and blockchain transaction monitoring."""
    metadata = {
        "name": "BlockchainAgent",
        "version": "1.0.0",
        "series": "b_series",
        "description":"Analyzes blockchain addresses, monitors transactions, and provides blockchain intelligence",
        "module": "agents.b_series.blockchain_agent"
    }
    def __init__(self):
        super().__init__(name=self.metadata["name"],
            description=self.metadata["description"])
        # Basic metadata for registry and discovery

        self.networks = {}
        self.contract_abis = {}

    async def setup(self):
        # Initialize supported blockchain networks
        self.networks = {
            "ethereum": {
                "explorer_url": "https://api.etherscan.io/api",
                "chain_id": 1,
                "currency": "ETH"
            },
            "binance": {
                "explorer_url": "https://api.bscscan.com/api",
                "chain_id": 56,
                "currency": "BNB"
            },
            "polygon": {
                "explorer_url": "https://api.polygonscan.com/api",
                "chain_id": 137,
                "currency": "MATIC"
            }
        }

        # Initialize common contract ABIs (simplified)
        self.contract_abis = {
            "erc20": ["transfer", "balanceOf", "approve", "totalSupply"],
            "erc721": ["transferFrom", "ownerOf", "approve", "setApprovalForAll"],
            "uniswap_v2": ["swapExactTokensForETH", "addLiquidity", "removeLiquidity"]
        }

        await asyncio.sleep(0.1)

    async def run(self, input_data: dict):
        try:
            # Validate input
            wallet_address = input_data.get("wallet_address", "")
            network = input_data.get("network", "ethereum")
            analysis_type = input_data.get("analysis_type", "wallet_overview")

            if not wallet_address:
                raise ValueError("No wallet address provided")

            # Validate wallet address format
            if not await self._validate_address_format(wallet_address, network):
                raise ValueError(f"Invalid {network} wallet address format")

            # Perform blockchain analysis based on type
            if analysis_type == "wallet_overview":
                analysis_result = await self._analyze_wallet_overview(wallet_address, network)
            elif analysis_type == "transaction_history":
                analysis_result = await self._analyze_transaction_history(wallet_address, network)
            elif analysis_type == "contract_interaction":
                analysis_result = await self._analyze_contract_interactions(wallet_address, network)
            else:
                analysis_result = await self._analyze_wallet_overview(wallet_address, network)

            # Generate risk assessment
            risk_assessment = await self._assess_wallet_risk(wallet_address, analysis_result)

            # Create blockchain summary
            blockchain_summary = await self._create_blockchain_summary(analysis_result, risk_assessment)

            result = {
                "wallet_analysis": analysis_result,
                "risk_assessment": risk_assessment,
                "blockchain_summary": blockchain_summary,
                "network_info": self.networks.get(network, {}),
                "analysis_timestamp": datetime.now().isoformat(),
                "recommendations": await self._generate_blockchain_recommendations(analysis_result, risk_assessment)
            }

            return {
                "agent": self.name,
                "status": "ok",
                "summary": f"Blockchain analysis complete for {wallet_address} on {network}: {risk_assessment['risk_level']} risk level",
                "data": result
            }

        except Exception as e:
            return await self.on_error(str(e))

    async def _validate_address_format(self, address: str, network: str) -> bool:
        """Validate blockchain address format"""
        address = address.lower().strip()

        if network == "ethereum":
            # Basic Ethereum address validation (42 chars, starts with 0x)
            return len(address) == 42 and address.startswith("0x")
        elif network == "binance":
            # BSC uses same format as Ethereum
            return len(address) == 42 and address.startswith("0x")
        elif network == "polygon":
            # Polygon uses same format as Ethereum
            return len(address) == 42 and address.startswith("0x")
        else:
            # Default validation for unknown networks
            return len(address) >= 26 and len(address) <= 64

    async def _analyze_wallet_overview(self, wallet_address: str, network: str) -> Dict[str, Any]:
        """Analyze wallet overview and basic metrics"""
        # Simulate blockchain data fetching
        balance = await self._get_wallet_balance(wallet_address, network)
        transaction_count = await self._get_transaction_count(wallet_address, network)
        first_seen = await self._get_first_transaction_date(wallet_address, network)

        return {
            "wallet_address": wallet_address,
            "network": network,
            "current_balance": balance,
            "balance_usd": balance * await self._get_current_price(network),
            "total_transactions": transaction_count,
            "first_seen": first_seen,
            "wallet_age_days": (datetime.now() - first_seen).days if first_seen else 0,
            "activity_level": await self._calculate_activity_level(transaction_count, first_seen),
            "tokens_held": await self._get_wallet_tokens(wallet_address, network)
        }

    async def _analyze_transaction_history(self, wallet_address: str, network: str) -> Dict[str, Any]:
        """Analyze transaction history and patterns"""
        # Simulate transaction history analysis
        transactions = await self._get_recent_transactions(wallet_address, network)

        return {
            "recent_transactions": transactions,
            "transaction_patterns": await self._analyze_transaction_patterns(transactions),
            "volume_analysis": await self._analyze_transaction_volume(transactions),
            "counterparties": await self._identify_counterparties(transactions),
            "gas_usage": await self._analyze_gas_usage(transactions)
        }

    async def _analyze_contract_interactions(self, wallet_address: str, network: str) -> Dict[str, Any]:
        """Analyze smart contract interactions"""
        contract_interactions = await self._get_contract_interactions(wallet_address, network)

        return {
            "contracts_interacted": contract_interactions,
            "contract_types": await self._classify_contracts(contract_interactions),
            "interaction_frequency": await self._calculate_interaction_frequency(contract_interactions),
            "popular_contracts": await self._identify_popular_contracts(contract_interactions)
        }

    async def _assess_wallet_risk(self, wallet_address: str, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess wallet risk level"""
        risk_score = 0
        risk_factors = []

        wallet_data = analysis_result.get("wallet_analysis", {})
        transaction_data = analysis_result.get("transaction_analysis", {})

        # New wallet risk
        wallet_age = wallet_data.get("wallet_age_days", 0)
        if wallet_age < 30:
            risk_score += 20
            risk_factors.append("New wallet (less than 30 days old)")

        # Low activity risk
        activity_level = wallet_data.get("activity_level", "low")
        if activity_level == "low":
            risk_score += 15
            risk_factors.append("Low transaction activity")

        # High volume risk
        volume_analysis = transaction_data.get("volume_analysis", {})
        if volume_analysis.get("daily_volume", 0) > 10000:  # $10k+ daily
            risk_score += 25
            risk_factors.append("High transaction volume")

        # Contract interaction risk
        contract_interactions = analysis_result.get("contract_analysis", {}).get("contracts_interacted", [])
        if len(contract_interactions) > 10:
            risk_score += 10
            risk_factors.append("Multiple contract interactions")

        # Determine risk level
        if risk_score >= 50:
            risk_level = "high"
        elif risk_score >= 25:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "confidence": max(0, 100 - risk_score)
        }

    async def _create_blockchain_summary(self, analysis_result: Dict[str, Any], risk_assessment: Dict[str, Any]) -> \
    Dict[str, Any]:
        """Create comprehensive blockchain summary"""
        wallet_data = analysis_result.get("wallet_analysis", {})

        return {
            "wallet_health": await self._assess_wallet_health(wallet_data),
            "network_activity": wallet_data.get("activity_level", "unknown"),
            "financial_metrics": {
                "total_balance": wallet_data.get("balance_usd", 0),
                "estimated_net_worth": await self._estimate_net_worth(wallet_data),
                "transaction_velocity": await self._calculate_transaction_velocity(analysis_result)
            },
            "behavioral_analysis": await self._analyze_wallet_behavior(analysis_result),
            "security_indicators": await self._assess_security_indicators(wallet_data, risk_assessment)
        }

    # Mock data generation methods for demonstration
    async def _get_wallet_balance(self, wallet_address: str, network: str) -> float:
        """Get wallet balance (mock implementation)"""
        # In production, this would call blockchain RPC
        base_balance = float(hashlib.sha256(wallet_address.encode()).hexdigest()[:8], 16) / 100000000
        return round(base_balance, 6)

    async def _get_transaction_count(self, wallet_address: str, network: str) -> int:
        """Get transaction count (mock implementation)"""
        return int(hashlib.sha256(wallet_address.encode()).hexdigest()[:4], 16) % 1000

    async def _get_first_transaction_date(self, wallet_address: str, network: str) -> datetime:
        """Get first transaction date (mock implementation)"""
        days_ago = int(hashlib.sha256(wallet_address.encode()).hexdigest()[:4], 16) % 1000
        return datetime.now() - timedelta(days=days_ago)

    async def _get_current_price(self, network: str) -> float:
        """Get current cryptocurrency price (mock implementation)"""
        prices = {
            "ethereum": 2500.0,
            "binance": 300.0,
            "polygon": 0.8
        }
        return prices.get(network, 1.0)

    async def _calculate_activity_level(self, transaction_count: int, first_seen: datetime) -> str:
        """Calculate wallet activity level"""
        if not first_seen:
            return "unknown"

        days_active = (datetime.now() - first_seen).days
        if days_active == 0:
            return "new"

        daily_transactions = transaction_count / days_active

        if daily_transactions > 5:
            return "very_high"
        elif daily_transactions > 2:
            return "high"
        elif daily_transactions > 0.5:
            return "medium"
        else:
            return "low"

    async def _get_wallet_tokens(self, wallet_address: str, network: str) -> List[Dict[str, Any]]:
        """Get tokens held in wallet (mock implementation)"""
        common_tokens = {
            "ethereum": ["USDC", "USDT", "DAI", "UNI", "LINK"],
            "binance": ["BUSD", "CAKE", "ADA", "DOT", "XRP"],
            "polygon": ["USDC", "QUICK", "AAVE", "SUSHI", "CRV"]
        }

        tokens = common_tokens.get(network, [])
        return [
            {
                "symbol": token,
                "balance": round(random.uniform(0.1, 1000.0), 4),
                "value_usd": round(random.uniform(10, 5000), 2)
            }
            for token in tokens[:3]  # Return 3 random tokens
        ]

    async def _get_recent_transactions(self, wallet_address: str, network: str) -> List[Dict[str, Any]]:
        """Get recent transactions (mock implementation)"""
        return [
            {
                "hash": f"0x{hashlib.sha256((wallet_address + str(i)).encode()).hexdigest()[:64]}",
                "timestamp": datetime.now() - timedelta(hours=i * 2),
                "value": round(random.uniform(0.01, 5.0), 4),
                "to": f"0x{hashlib.sha256(str(i).encode()).hexdigest()[:40]}",
                "status": "confirmed",
                "gas_used": random.randint(21000, 100000)
            }
            for i in range(5)  # Last 5 transactions
        ]

    async def _analyze_transaction_patterns(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze transaction patterns"""
        if not transactions:
            return {}

        values = [tx["value"] for tx in transactions]
        times = [tx["timestamp"] for tx in transactions]

        return {
            "average_value": statistics.mean(values) if values else 0,
            "value_std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "time_pattern": await self._detect_time_pattern(times),
            "transaction_frequency": len(transactions) / 24  # per hour (24h window)
        }

    async def _analyze_transaction_volume(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze transaction volume"""
        total_volume = sum(tx["value"] for tx in transactions)

        return {
            "total_volume": total_volume,
            "daily_volume": total_volume * 24,  # Extrapolate
            "largest_transaction": max(tx["value"] for tx in transactions) if transactions else 0,
            "volume_trend": "increasing" if len(transactions) > 3 else "stable"
        }

    async def _identify_counterparties(self, transactions: List[Dict[str, Any]]) -> List[str]:
        """Identify frequent counterparties"""
        counterparties = [tx["to"] for tx in transactions]
        # Return unique counterparties
        return list(set(counterparties))[:5]

    async def _analyze_gas_usage(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze gas usage patterns"""
        if not transactions:
            return {}

        gas_values = [tx["gas_used"] for tx in transactions]

        return {
            "average_gas": statistics.mean(gas_values),
            "total_gas": sum(gas_values),
            "gas_efficiency": "efficient" if statistics.mean(gas_values) < 50000 else "standard"
        }

    async def _get_contract_interactions(self, wallet_address: str, network: str) -> List[Dict[str, Any]]:
        """Get contract interactions (mock implementation)"""
        common_contracts = [
            {"address": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D", "name": "Uniswap V2 Router", "type": "dex"},
            {"address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", "name": "USDC Token", "type": "erc20"},
            {"address": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984", "name": "Uniswap Token", "type": "erc20"}
        ]

        return [
            {**contract, "interaction_count": random.randint(1, 20)}
            for contract in common_contracts[:2]  # 2 random contracts
        ]

    async def _classify_contracts(self, contracts: List[Dict[str, Any]]) -> Dict[str, int]:
        """Classify contracts by type"""
        type_count = {}
        for contract in contracts:
            contract_type = contract.get("type", "unknown")
            type_count[contract_type] = type_count.get(contract_type, 0) + 1
        return type_count

    async def _calculate_interaction_frequency(self, contracts: List[Dict[str, Any]]) -> str:
        """Calculate contract interaction frequency"""
        total_interactions = sum(contract.get("interaction_count", 0) for contract in contracts)

        if total_interactions > 50:
            return "very_high"
        elif total_interactions > 20:
            return "high"
        elif total_interactions > 5:
            return "medium"
        else:
            return "low"

    async def _identify_popular_contracts(self, contracts: List[Dict[str, Any]]) -> List[str]:
        """Identify most popular contracts by interaction count"""
        sorted_contracts = sorted(contracts, key=lambda x: x.get("interaction_count", 0), reverse=True)
        return [contract["name"] for contract in sorted_contracts[:3]]

    async def _assess_wallet_health(self, wallet_data: Dict[str, Any]) -> str:
        """Assess overall wallet health"""
        balance = wallet_data.get("balance_usd", 0)
        activity = wallet_data.get("activity_level", "low")

        if balance > 1000 and activity in ["high", "very_high"]:
            return "excellent"
        elif balance > 100 and activity in ["medium", "high"]:
            return "good"
        elif balance > 10:
            return "fair"
        else:
            return "poor"

    async def _estimate_net_worth(self, wallet_data: Dict[str, Any]) -> float:
        """Estimate wallet net worth"""
        base_balance = wallet_data.get("balance_usd", 0)
        tokens = wallet_data.get("tokens_held", [])
        token_value = sum(token.get("value_usd", 0) for token in tokens)

        return base_balance + token_value

    async def _calculate_transaction_velocity(self, analysis_result: Dict[str, Any]) -> float:
        """Calculate transaction velocity"""
        transaction_data = analysis_result.get("transaction_analysis", {})
        volume_analysis = transaction_data.get("volume_analysis", {})

        return volume_analysis.get("daily_volume", 0) / 24  # Hourly velocity

    async def _analyze_wallet_behavior(self, analysis_result: Dict[str, Any]) -> Dict[str, str]:
        """Analyze wallet behavior patterns"""
        wallet_data = analysis_result.get("wallet_analysis", {})
        activity = wallet_data.get("activity_level", "low")

        behaviors = []
        if activity == "very_high":
            behaviors.append("Active trader")
        elif activity == "high":
            behaviors.append("Regular user")
        elif activity == "medium":
            behaviors.append("Occasional user")
        else:
            behaviors.append("Dormant/Low activity")

        # Add behavior based on contract interactions
        contract_analysis = analysis_result.get("contract_analysis", {})
        if contract_analysis.get("contracts_interacted"):
            behaviors.append("DeFi user")

        return {
            "primary_behavior": behaviors[0] if behaviors else "Unknown",
            "behavior_tags": behaviors
        }

    async def _assess_security_indicators(self, wallet_data: Dict[str, Any], risk_assessment: Dict[str, Any]) -> Dict[
        str, bool]:
        """Assess wallet security indicators"""
        wallet_age = wallet_data.get("wallet_age_days", 0)
        activity = wallet_data.get("activity_level", "low")

        return {
            "established_wallet": wallet_age > 180,  # 6+ months
            "consistent_activity": activity in ["medium", "high", "very_high"],
            "diversified_holdings": len(wallet_data.get("tokens_held", [])) > 1,
            "low_risk_profile": risk_assessment.get("risk_level") == "low"
        }

    async def _generate_blockchain_recommendations(self, analysis_result: Dict[str, Any],
                                                   risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate blockchain-specific recommendations"""
        recommendations = []
        risk_level = risk_assessment.get("risk_level", "low")

        if risk_level == "high":
            recommendations.append("Monitor wallet activity closely for suspicious transactions")
            recommendations.append("Consider using hardware wallet for enhanced security")
        elif risk_level == "medium":
            recommendations.append("Regular security audit recommended")
            recommendations.append("Enable transaction signing confirmations")

        wallet_data = analysis_result.get("wallet_analysis", {})
        if wallet_data.get("balance_usd", 0) > 1000:
            recommendations.append("Consider diversifying assets across multiple wallets")

        if not recommendations:
            recommendations.append("Wallet appears secure. Continue standard security practices.")

        return recommendations

    async def _detect_time_pattern(self, timestamps: List[datetime]) -> str:
        """Detect transaction time patterns"""
        if len(timestamps) < 3:
            return "insufficient_data"

        hours = [ts.hour for ts in timestamps]
        unique_hours = len(set(hours))

        if unique_hours / len(hours) > 0.7:
            return "distributed"
        else:
            return "clustered"
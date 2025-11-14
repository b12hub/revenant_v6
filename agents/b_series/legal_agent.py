# /agents/b_series/legal_agent.py
from core.agent_base import RevenantAgentBase
import asyncio
import re
from typing import Dict, List, Any
from datetime import datetime


class LegalAgent(RevenantAgentBase):
    """Contract parsing, GDPR & compliance checks using regex patterns and legal keyword analysis."""
    metadata = {
        "name": "LegalAgent",
        "version": "1.0.0",
        "series": "b_series",
        "description": "Analyzes legal documents for compliance issues, GDPR requirements, and contract terms",
        "module": "agents.b_series.legal_agent"
    }
    def __init__(self):
        super().__init__(name="LegalAgent",
            description=self.metadata["description"]
        )
        self.compliance_patterns = {}
        self.gdpr_keywords = {}

    async def setup(self):
        # Initialize legal patterns and keywords
        self.compliance_patterns = {
            "liability_limitation": [
                r"limitation of liability",
                r"maximum liability",
                r"liability cap",
                r"not liable for",
                r"exclusion of liability"
            ],
            "confidentiality": [
                r"confidential information",
                r"non-disclosure",
                r"proprietary information",
                r"trade secrets"
            ],
            "termination": [
                r"termination for cause",
                r"termination without cause",
                r"notice period",
                r"termination clause"
            ],
            "governing_law": [
                r"governing law",
                r"jurisdiction",
                r"venue",
                r"dispute resolution"
            ]
        }

        self.gdpr_keywords = {
            "data_processing": ["data processing", "data controller", "data processor", "personal data"],
            "consent": ["consent", "opt-in", "opt-out", "explicit consent"],
            "rights": ["right to access", "right to erasure", "right to rectification", "data portability"],
            "security": ["data security", "breach notification", "encryption", "access controls"],
            "transfer": ["data transfer", "third country", "adequate protection", "binding corporate rules"]
        }

        await asyncio.sleep(0.1)

    async def run(self, input_data: dict):
        try:
            # Validate input
            contract_text = input_data.get("contract", "")
            if not contract_text:
                raise ValueError("No contract text provided")

            # Analyze contract clauses
            clause_analysis = await self._analyze_contract_clauses(contract_text)

            # Check GDPR compliance
            gdpr_analysis = await self._check_gdpr_compliance(contract_text)

            # Identify potential risks
            risk_assessment = await self._assess_legal_risks(contract_text, clause_analysis)

            # Generate compliance score
            compliance_score = await self._calculate_compliance_score(clause_analysis, gdpr_analysis)

            result = {
                "document_metadata": {
                    "word_count": len(contract_text.split()),
                    "analysis_timestamp": datetime.now().isoformat(),
                    "clauses_identified": len(clause_analysis["clauses_found"])
                },
                "clause_analysis": clause_analysis,
                "gdpr_compliance": gdpr_analysis,
                "risk_assessment": risk_assessment,
                "compliance_score": compliance_score,
                "recommendations": await self._generate_recommendations(clause_analysis, gdpr_analysis, risk_assessment)
            }

            return {
                "agent": self.name,
                "status": "ok",
                "summary": f"Legal analysis complete: {compliance_score}/100 compliance score, {len(risk_assessment['high_risks'])} high risks identified",
                "data": result
            }

        except Exception as e:
            return await self.on_error(str(e))

    async def _analyze_contract_clauses(self, contract_text: str) -> Dict[str, Any]:
        """Analyze contract text for common legal clauses"""
        clauses_found = {}
        text_lower = contract_text.lower()

        for clause_type, patterns in self.compliance_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text_lower)
                if found:
                    matches.extend(found)

            if matches:
                clauses_found[clause_type] = {
                    "count": len(matches),
                    "examples": list(set(matches))[:3],  # Unique examples
                    "risk_level": await self._assess_clause_risk(clause_type, matches)
                }

        return {
            "clauses_found": clauses_found,
            "total_clauses_identified": sum(len(data.get("examples", [])) for data in clauses_found.values()),
            "coverage_score": len(clauses_found) / len(self.compliance_patterns) if self.compliance_patterns else 0
        }

    async def _check_gdpr_compliance(self, contract_text: str) -> Dict[str, Any]:
        """Check GDPR compliance requirements"""
        text_lower = contract_text.lower()
        gdpr_checks = {}

        for category, keywords in self.gdpr_keywords.items():
            found_keywords = []
            for keyword in keywords:
                if keyword in text_lower:
                    found_keywords.append(keyword)

            gdpr_checks[category] = {
                "keywords_found": found_keywords,
                "compliance_status": "compliant" if found_keywords else "missing",
                "importance": await self._get_gdpr_importance(category)
            }

        missing_requirements = [cat for cat, data in gdpr_checks.items() if data["compliance_status"] == "missing"]

        return {
            "gdpr_checks": gdpr_checks,
            "missing_requirements": missing_requirements,
            "compliance_level": await self._calculate_gdpr_compliance_level(gdpr_checks),
            "recommended_actions": await self._suggest_gdpr_improvements(missing_requirements)
        }

    async def _assess_legal_risks(self, contract_text: str, clause_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess potential legal risks in the contract"""
        risks = {
            "high_risks": [],
            "medium_risks": [],
            "low_risks": []
        }

        text_lower = contract_text.lower()

        # Check for ambiguous language
        ambiguous_terms = ["reasonable", "best efforts", "as soon as practicable", "material adverse effect"]
        found_ambiguous = [term for term in ambiguous_terms if term in text_lower]
        if found_ambiguous:
            risks["medium_risks"].append({
                "type": "ambiguous_language",
                "description": f"Ambiguous terms found: {', '.join(found_ambiguous)}",
                "suggestion": "Define ambiguous terms more precisely"
            })

        # Check for one-sided clauses
        one_sided_indicators = [
            "sole discretion", "unilateral", "at our discretion", "company may"
        ]
        one_sided_count = sum(1 for indicator in one_sided_indicators if indicator in text_lower)
        if one_sided_count > 2:
            risks["high_risks"].append({
                "type": "one_sided_terms",
                "description": f"Multiple one-sided terms detected ({one_sided_count})",
                "suggestion": "Review for balanced terms and mutual obligations"
            })

        # Check liability limitations
        liability_patterns = ["limitation of liability", "liability cap", "maximum liability"]
        liability_found = any(pattern in text_lower for pattern in liability_patterns)
        if liability_found:
            risks["medium_risks"].append({
                "type": "liability_limitation",
                "description": "Liability limitations present",
                "suggestion": "Ensure liability caps are reasonable and insurable"
            })

        return risks

    async def _calculate_compliance_score(self, clause_analysis: Dict[str, Any],
                                          gdpr_analysis: Dict[str, Any]) -> float:
        """Calculate overall compliance score (0-100)"""
        base_score = 50.0

        # Adjust based on clause coverage
        coverage_score = clause_analysis.get("coverage_score", 0) * 20
        base_score += coverage_score

        # Adjust based on GDPR compliance
        gdpr_level = gdpr_analysis.get("compliance_level", 0)
        base_score += gdpr_level * 30

        # Ensure score is within bounds
        return max(0.0, min(100.0, base_score))

    async def _generate_recommendations(self, clause_analysis: Dict[str, Any], gdpr_analysis: Dict[str, Any],
                                        risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate legal recommendations"""
        recommendations = []

        # Clause coverage recommendations
        coverage_score = clause_analysis.get("coverage_score", 0)
        if coverage_score < 0.7:
            recommendations.append("Consider adding missing standard contract clauses")

        # GDPR recommendations
        missing_gdpr = gdpr_analysis.get("missing_requirements", [])
        if missing_gdpr:
            recommendations.append(f"Add GDPR requirements for: {', '.join(missing_gdpr[:3])}")

        # Risk-based recommendations
        high_risks = risk_assessment.get("high_risks", [])
        if high_risks:
            recommendations.append("Address high-risk items identified in risk assessment")

        # General recommendations
        if not recommendations:
            recommendations.append(
                "Contract appears reasonably comprehensive. Review with legal counsel for specific use case.")
        else:
            recommendations.append("Consult with legal professional before finalizing contract")

        return recommendations

    async def _assess_clause_risk(self, clause_type: str, matches: List[str]) -> str:
        """Assess risk level for a clause type"""
        high_risk_clauses = ["liability_limitation", "termination"]
        medium_risk_clauses = ["confidentiality", "governing_law"]

        if clause_type in high_risk_clauses:
            return "high"
        elif clause_type in medium_risk_clauses:
            return "medium"
        else:
            return "low"

    async def _get_gdpr_importance(self, category: str) -> str:
        """Get importance level for GDPR categories"""
        critical_categories = ["data_processing", "consent", "security"]
        important_categories = ["rights", "transfer"]

        if category in critical_categories:
            return "critical"
        elif category in important_categories:
            return "important"
        else:
            return "standard"

    async def _calculate_gdpr_compliance_level(self, gdpr_checks: Dict[str, Any]) -> float:
        """Calculate GDPR compliance level (0-1)"""
        total_categories = len(gdpr_checks)
        if total_categories == 0:
            return 0.0

        compliant_categories = sum(1 for check in gdpr_checks.values() if check["compliance_status"] == "compliant")
        return compliant_categories / total_categories

    async def _suggest_gdpr_improvements(self, missing_requirements: List[str]) -> List[str]:
        """Suggest improvements for missing GDPR requirements"""
        suggestions = []

        requirement_mapping = {
            "data_processing": "Include data processing agreements and define roles of data controller/processor",
            "consent": "Add explicit consent mechanisms and opt-in/opt-out procedures",
            "rights": "Outline data subject rights and procedures for exercising them",
            "security": "Describe data security measures and breach notification procedures",
            "transfer": "Specify conditions for international data transfers"
        }

        for requirement in missing_requirements:
            if requirement in requirement_mapping:
                suggestions.append(requirement_mapping[requirement])

        return suggestions
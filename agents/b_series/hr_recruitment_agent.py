# /agents/b_series/hr_recruitment_agent.py
from core.agent_base import RevenantAgentBase
import asyncio
from typing import Dict, List, Any
from datetime import datetime, timedelta
import statistics


class HRRecruitmentAgent(RevenantAgentBase):
    """Candidate ranking and interview scheduling using skills matching and availability coordination."""
    metadata = {
        "name": "HRRecruitmentAgent",
        "version": "1.0.0",
        "series": "b_series",
        "description": "Evaluates candidates, ranks them based on qualifications, and schedules interview timelines",
        "module": "agents.b_series.hr_recruitment_agent"
    }
    def __init__(self):
        super().__init__(name=self.metadata["name"],
            description=self.metadata["description"])
        self.skill_weights = {}
        self.interview_calendar = {}

    async def setup(self):
        # Initialize skill importance weights
        self.skill_weights = {
            "technical_skills": 0.35,
            "experience": 0.25,
            "education": 0.15,
            "soft_skills": 0.15,
            "cultural_fit": 0.10
        }

        # Initialize interview calendar (mock data)
        self.interview_calendar = {
            "available_slots": [
                datetime.now() + timedelta(days=1, hours=9),
                datetime.now() + timedelta(days=1, hours=11),
                datetime.now() + timedelta(days=2, hours=10),
                datetime.now() + timedelta(days=2, hours=14),
                datetime.now() + timedelta(days=3, hours=9)
            ],
            "booked_slots": []
        }

        await asyncio.sleep(0.1)

    async def run(self, input_data: dict):
        try:
            # Validate input
            candidates = input_data.get("candidates", [])
            job_requirements = input_data.get("job_requirements", {})

            if not candidates:
                raise ValueError("No candidates provided for evaluation")

            # Evaluate and rank candidates
            ranked_candidates = await self._evaluate_candidates(candidates, job_requirements)

            # Schedule interviews for top candidates
            interview_schedule = await self._schedule_interviews(ranked_candidates[:3])

            # Generate hiring recommendations
            hiring_recommendations = await self._generate_hiring_recommendations(ranked_candidates)

            result = {
                "evaluation_metrics": {
                    "total_candidates": len(candidates),
                    "evaluation_criteria": list(self.skill_weights.keys()),
                    "top_score": ranked_candidates[0]["total_score"] if ranked_candidates else 0
                },
                "ranked_candidates": ranked_candidates,
                "top_candidates": ranked_candidates[:3],
                "interview_schedule": interview_schedule,
                "hiring_recommendations": hiring_recommendations,
                "candidate_comparison": await self._compare_candidates(ranked_candidates[:5])
            }

            return {
                "agent": self.name,
                "status": "ok",
                "summary": f"Evaluated {len(candidates)} candidates, identified top 3 with scores {[c['total_score'] for c in ranked_candidates[:3]]}",
                "data": result
            }

        except Exception as e:
            return await self.on_error(str(e))

    async def _evaluate_candidates(self, candidates: List[Dict], job_requirements: Dict) -> List[Dict]:
        """Evaluate and rank candidates based on job requirements"""
        evaluated_candidates = []

        for candidate in candidates:
            candidate_score = await self._calculate_candidate_score(candidate, job_requirements)
            evaluated_candidate = {
                **candidate,
                "total_score": candidate_score,
                "skill_breakdown": await self._calculate_skill_breakdown(candidate, job_requirements),
                "strengths": await self._identify_strengths(candidate, job_requirements),
                "weaknesses": await self._identify_weaknesses(candidate, job_requirements),
                "recommendation": await self._generate_candidate_recommendation(candidate_score)
            }
            evaluated_candidates.append(evaluated_candidate)

        # Sort candidates by total score (descending)
        return sorted(evaluated_candidates, key=lambda x: x["total_score"], reverse=True)

    async def _calculate_candidate_score(self, candidate: Dict, job_requirements: Dict) -> float:
        """Calculate overall candidate score (0-100)"""
        total_score = 0.0

        # Technical skills evaluation
        tech_skills = candidate.get("skills", [])
        required_skills = job_requirements.get("required_skills", [])
        tech_score = await self._calculate_skill_match(tech_skills, required_skills)
        total_score += tech_score * self.skill_weights["technical_skills"]

        # Experience evaluation
        experience_years = candidate.get("experience_years", 0)
        required_experience = job_requirements.get("required_experience", 0)
        exp_score = min(experience_years / max(required_experience, 1), 1.5)  # Cap at 1.5x requirement
        total_score += exp_score * self.skill_weights["experience"]

        # Education evaluation
        education_level = candidate.get("education_level", "").lower()
        required_education = job_requirements.get("required_education", "").lower()
        edu_score = await self._calculate_education_score(education_level, required_education)
        total_score += edu_score * self.skill_weights["education"]

        # Soft skills evaluation
        soft_skills = candidate.get("soft_skills", [])
        required_soft_skills = job_requirements.get("required_soft_skills", [])
        soft_score = await self._calculate_skill_match(soft_skills, required_soft_skills)
        total_score += soft_score * self.skill_weights["soft_skills"]

        # Cultural fit (simplified)
        cultural_fit = candidate.get("cultural_fit_score", 0.5)
        total_score += cultural_fit * self.skill_weights["cultural_fit"]

        return min(100.0, total_score * 100)  # Convert to percentage

    async def _calculate_skill_match(self, candidate_skills: List[str], required_skills: List[str]) -> float:
        """Calculate skill matching score"""
        if not required_skills:
            return 0.7  # Default score if no requirements specified

        candidate_skills_lower = [skill.lower() for skill in candidate_skills]
        required_skills_lower = [skill.lower() for skill in required_skills]

        matched_skills = sum(1 for skill in required_skills_lower if skill in candidate_skills_lower)
        return matched_skills / len(required_skills_lower)

    async def _calculate_education_score(self, candidate_education: str, required_education: str) -> float:
        """Calculate education level score"""
        education_hierarchy = {
            "high school": 1,
            "associate": 2,
            "bachelor": 3,
            "master": 4,
            "phd": 5
        }

        candidate_level = education_hierarchy.get(candidate_education, 0)
        required_level = education_hierarchy.get(required_education, 0)

        if candidate_level >= required_level:
            return 1.0
        elif required_level > 0:
            return candidate_level / required_level
        else:
            return 0.5  # Neutral score if no requirement

    async def _calculate_skill_breakdown(self, candidate: Dict, job_requirements: Dict) -> Dict[str, float]:
        """Calculate detailed skill breakdown"""
        return {
            "technical_skills": await self._calculate_skill_match(
                candidate.get("skills", []),
                job_requirements.get("required_skills", [])
            ),
            "experience": min(
                candidate.get("experience_years", 0) / max(job_requirements.get("required_experience", 1), 1), 1.5),
            "education": await self._calculate_education_score(
                candidate.get("education_level", ""),
                job_requirements.get("required_education", "")
            ),
            "soft_skills": await self._calculate_skill_match(
                candidate.get("soft_skills", []),
                job_requirements.get("required_soft_skills", [])
            ),
            "cultural_fit": candidate.get("cultural_fit_score", 0.5)
        }

    async def _identify_strengths(self, candidate: Dict, job_requirements: Dict) -> List[str]:
        """Identify candidate strengths"""
        strengths = []

        # Technical skills strength
        candidate_skills = candidate.get("skills", [])
        required_skills = job_requirements.get("required_skills", [])
        if len(candidate_skills) > len(required_skills):
            strengths.append("Broad technical skill set beyond requirements")

        # Experience strength
        experience_years = candidate.get("experience_years", 0)
        required_experience = job_requirements.get("required_experience", 0)
        if experience_years > required_experience:
            strengths.append(f"Extensive experience ({experience_years} years)")

        # Education strength
        education_level = candidate.get("education_level", "").lower()
        if "phd" in education_level or "master" in education_level:
            strengths.append("Advanced degree holder")

        return strengths[:3]  # Return top 3 strengths

    async def _identify_weaknesses(self, candidate: Dict, job_requirements: Dict) -> List[str]:
        """Identify candidate weaknesses"""
        weaknesses = []

        # Missing required skills
        candidate_skills = [s.lower() for s in candidate.get("skills", [])]
        required_skills = [s.lower() for s in job_requirements.get("required_skills", [])]
        missing_skills = [skill for skill in required_skills if skill not in candidate_skills]
        if missing_skills:
            weaknesses.append(f"Missing required skills: {', '.join(missing_skills[:2])}")

        # Experience gap
        experience_years = candidate.get("experience_years", 0)
        required_experience = job_requirements.get("required_experience", 0)
        if experience_years < required_experience:
            weaknesses.append(f"Below required experience ({experience_years}/{required_experience} years)")

        return weaknesses[:2]  # Return top 2 weaknesses

    async def _generate_candidate_recommendation(self, score: float) -> str:
        """Generate hiring recommendation based on score"""
        if score >= 80:
            return "Strongly Recommend - Schedule final interview"
        elif score >= 65:
            return "Recommend - Proceed to technical interview"
        elif score >= 50:
            return "Consider - Initial screening recommended"
        else:
            return "Not Recommended - Does not meet minimum requirements"

    async def _schedule_interviews(self, top_candidates: List[Dict]) -> Dict[str, Any]:
        """Schedule interviews for top candidates"""
        scheduled_interviews = []

        for i, candidate in enumerate(top_candidates):
            if i < len(self.interview_calendar["available_slots"]):
                slot = self.interview_calendar["available_slots"][i]
                interview = {
                    "candidate_id": candidate.get("id", f"candidate_{i}"),
                    "candidate_name": candidate.get("name", "Unknown"),
                    "interview_slot": slot.isoformat(),
                    "interview_type": "Technical" if i == 0 else "Cultural Fit",
                    "duration_minutes": 60,
                    "interviewers": await self._assign_interviewers(candidate)
                }
                scheduled_interviews.append(interview)

        return {
            "scheduled_interviews": scheduled_interviews,
            "available_slots_remaining": len(self.interview_calendar["available_slots"]) - len(scheduled_interviews),
            "scheduling_timeline": await self._generate_scheduling_timeline(scheduled_interviews)
        }

    async def _assign_interviewers(self, candidate: Dict) -> List[str]:
        """Assign appropriate interviewers based on candidate profile"""
        base_interviewers = ["HR Manager", "Hiring Manager"]

        # Add technical interviewer for technical roles
        if candidate.get("skills"):
            base_interviewers.append("Technical Lead")

        # Add department head for senior positions
        if candidate.get("experience_years", 0) > 5:
            base_interviewers.append("Department Head")

        return base_interviewers

    async def _generate_scheduling_timeline(self, scheduled_interviews: List[Dict]) -> Dict[str, Any]:
        """Generate interview scheduling timeline"""
        if not scheduled_interviews:
            return {}

        interview_dates = [datetime.fromisoformat(interview["interview_slot"]) for interview in scheduled_interviews]

        return {
            "first_interview": min(interview_dates).isoformat(),
            "last_interview": max(interview_dates).isoformat(),
            "total_duration_days": (max(interview_dates) - min(interview_dates)).days + 1,
            "interview_frequency": "daily" if len(interview_dates) > 1 else "single"
        }

    async def _generate_hiring_recommendations(self, ranked_candidates: List[Dict]) -> Dict[str, Any]:
        """Generate overall hiring recommendations"""
        if not ranked_candidates:
            return {"recommendation": "No suitable candidates found"}

        top_candidate = ranked_candidates[0]
        scores = [candidate["total_score"] for candidate in ranked_candidates[:5]]

        return {
            "primary_recommendation": f"Hire {top_candidate.get('name', 'top candidate')} with score {top_candidate['total_score']:.1f}",
            "backup_candidates": [candidate.get("name", f"Candidate {i}") for i, candidate in
                                  enumerate(ranked_candidates[1:4], 2)],
            "score_distribution": {
                "average": statistics.mean(scores) if scores else 0,
                "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0,
                "high_score": max(scores) if scores else 0,
                "low_score": min(scores) if scores else 0
            },
            "hiring_confidence": await self._calculate_hiring_confidence(top_candidate["total_score"])
        }

    async def _calculate_hiring_confidence(self, top_score: float) -> str:
        """Calculate hiring confidence level"""
        if top_score >= 85:
            return "Very High"
        elif top_score >= 70:
            return "High"
        elif top_score >= 60:
            return "Moderate"
        else:
            return "Low"

    async def _compare_candidates(self, top_candidates: List[Dict]) -> Dict[str, Any]:
        """Compare top candidates across key dimensions"""
        if len(top_candidates) < 2:
            return {"message": "Insufficient candidates for comparison"}

        comparison = {
            "technical_skills_ranking": [],
            "experience_ranking": [],
            "overall_ranking": []
        }

        # Technical skills ranking
        tech_ranking = sorted(top_candidates,
                              key=lambda x: x["skill_breakdown"]["technical_skills"],
                              reverse=True)
        comparison["technical_skills_ranking"] = [candidate.get("name", "Unknown") for candidate in tech_ranking]

        # Experience ranking
        exp_ranking = sorted(top_candidates,
                             key=lambda x: x["skill_breakdown"]["experience"],
                             reverse=True)
        comparison["experience_ranking"] = [candidate.get("name", "Unknown") for candidate in exp_ranking]

        # Overall ranking (already sorted by total_score)
        comparison["overall_ranking"] = [candidate.get("name", "Unknown") for candidate in top_candidates]

        return comparison
# /agents/c_series/research_agent.py
import statistics

from core.agent_base import RevenantAgentBase
import asyncio
from typing import Dict, List, Any
from datetime import datetime
import re
from collections import Counter


class ResearchAgent(RevenantAgentBase):
    """Perform contextual research synthesis across data and documents with summarization, trend extraction, and knowledge contextualization."""
    metadata = {
        "name": "ResearchAgent",
        "version": "1.0.0",
        "series": "c_series",
        "description": "Synthesizes research from multiple sources, extracts trends, and contextualizes knowledge for decision support",
        "module": "agents.c_series.research_agent"
    }

    def __init__(self):
        super().__init__(
            name=self.metadata['name'],
            description=self.metadata['description'])
        self.knowledge_base = {}
        self.research_methods = {}
        self.trend_patterns = {}

    async def setup(self):
        # Initialize knowledge base and research methods
        self.knowledge_base = {
            "documents": {},
            "summaries": {},
            "key_findings": {},
            "citations": {}
        }

        self.research_methods = {
            "summarization": {
                "extractive": {"chunk_size": 500, "compression_ratio": 0.3},
                "abstractive": {"max_length": 200, "temperature": 0.7}
            },
            "trend_analysis": {
                "temporal_patterns": True,
                "topic_evolution": True,
                "sentiment_tracking": True
            },
            "synthesis": {
                "cross_referencing": True,
                "contradiction_detection": True,
                "knowledge_gap_identification": True
            }
        }

        self.trend_patterns = {
            "emerging_topics": [],
            "declining_topics": [],
            "stable_topics": [],
            "controversial_topics": []
        }

        await asyncio.sleep(0.1)

    async def run(self, input_data: dict):
        try:
            # Validate input
            research_materials = input_data.get("research_materials", [])
            research_focus = input_data.get("research_focus", "general")
            synthesis_depth = input_data.get("synthesis_depth", "comprehensive")

            if not research_materials:
                raise ValueError("No research materials provided")

            # Process research materials
            processed_materials = await self._process_research_materials(research_materials)

            # Perform content analysis
            content_analysis = await self._analyze_content(processed_materials, research_focus)

            # Extract key trends and patterns
            trend_analysis = await self._analyze_trends(processed_materials, research_focus)

            # Synthesize findings
            research_synthesis = await self._synthesize_findings(processed_materials, content_analysis, trend_analysis,
                                                                 synthesis_depth)

            # Generate research report
            research_report = await self._generate_research_report(research_synthesis, research_focus)

            # Update knowledge base
            knowledge_update = await self._update_knowledge_base(research_synthesis, research_focus)

            result = {
                "research_scope": {
                    "materials_processed": len(research_materials),
                    "research_focus": research_focus,
                    "synthesis_depth": synthesis_depth,
                    "analysis_timestamp": datetime.now().isoformat()
                },
                "content_analysis": content_analysis,
                "trend_analysis": trend_analysis,
                "research_synthesis": research_synthesis,
                "research_report": research_report,
                "knowledge_update": knowledge_update,
                "actionable_insights": await self._extract_actionable_insights(research_synthesis),
                "research_quality": await self._assess_research_quality(processed_materials, research_synthesis)
            }

            return {
                "agent": self.name,
                "status": "ok",
                "summary": f"Research synthesis complete: Analyzed {len(research_materials)} sources, identified {len(trend_analysis['key_trends'])} key trends, generated {len(research_synthesis['key_findings'])} findings",
                "data": result
            }

        except Exception as e:
            return await self.on_error(str(e))

    async def _process_research_materials(self, materials: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process and normalize research materials"""
        processed_documents = []
        metadata_analysis = {}

        for i, material in enumerate(materials):
            document = await self._process_single_document(material, i)
            processed_documents.append(document)

        # Analyze overall metadata
        metadata_analysis = await self._analyze_collection_metadata(processed_documents)

        return {
            "documents": processed_documents,
            "collection_metadata": metadata_analysis,
            "processing_metrics": {
                "total_documents": len(processed_documents),
                "total_content_length": sum(doc.get("content_length", 0) for doc in processed_documents),
                "content_types": list(set(doc.get("content_type", "unknown") for doc in processed_documents))
            }
        }

    async def _process_single_document(self, material: Dict[str, Any], doc_id: int) -> Dict[str, Any]:
        """Process a single research document"""
        content = material.get("content", "")
        metadata = material.get("metadata", {})

        # Extract key information
        key_phrases = await self._extract_key_phrases(content)
        entities = await self._extract_entities(content)
        sentiment = await self._analyze_sentiment(content)

        return {
            "doc_id": f"doc_{doc_id}",
            "content": content,
            "content_length": len(content),
            "content_type": metadata.get("type", "text"),
            "metadata": metadata,
            "key_phrases": key_phrases,
            "entities": entities,
            "sentiment": sentiment,
            "readability_score": await self._assess_readability(content),
            "information_density": await self._calculate_information_density(content, key_phrases),
            "processed_timestamp": datetime.now().isoformat()
        }

    async def _analyze_content(self, processed_materials: Dict[str, Any], research_focus: str) -> Dict[str, Any]:
        """Analyze content across research materials"""
        documents = processed_materials["documents"]

        # Thematic analysis
        themes = await self._identify_themes(documents, research_focus)

        # Content quality assessment
        quality_metrics = await self._assess_content_quality(documents)

        # Information extraction
        key_information = await self._extract_key_information(documents, research_focus)

        return {
            "thematic_analysis": themes,
            "quality_metrics": quality_metrics,
            "key_information": key_information,
            "content_coverage": await self._assess_content_coverage(documents, research_focus),
            "source_reliability": await self._assess_source_reliability(documents)
        }

    async def _analyze_trends(self, processed_materials: Dict[str, Any], research_focus: str) -> Dict[str, Any]:
        """Analyze trends across research materials"""
        documents = processed_materials["documents"]

        # Temporal trends (if timestamps available)
        temporal_trends = await self._analyze_temporal_trends(documents)

        # Topic evolution
        topic_evolution = await self._analyze_topic_evolution(documents)

        # Sentiment trends
        sentiment_trends = await self._analyze_sentiment_trends(documents)

        # Emerging patterns
        emerging_patterns = await self._identify_emerging_patterns(documents, research_focus)

        return {
            "temporal_trends": temporal_trends,
            "topic_evolution": topic_evolution,
            "sentiment_trends": sentiment_trends,
            "emerging_patterns": emerging_patterns,
            "key_trends": await self._synthesize_key_trends(temporal_trends, topic_evolution, sentiment_trends,
                                                            emerging_patterns),
            "trend_confidence": await self._calculate_trend_confidence(documents)
        }

    async def _synthesize_findings(self, processed_materials: Dict[str, Any], content_analysis: Dict[str, Any],
                                   trend_analysis: Dict[str, Any], synthesis_depth: str) -> Dict[str, Any]:
        """Synthesize research findings"""
        documents = processed_materials["documents"]

        # Generate comprehensive summary
        comprehensive_summary = await self._generate_comprehensive_summary(documents, content_analysis, trend_analysis)

        # Extract key findings
        key_findings = await self._extract_key_findings(documents, content_analysis, trend_analysis)

        # Identify knowledge gaps
        knowledge_gaps = await self._identify_knowledge_gaps(documents, content_analysis, trend_analysis)

        # Detect contradictions and consensus
        contradiction_analysis = await self._analyze_contradictions(documents)

        # Generate recommendations
        recommendations = await self._generate_recommendations(key_findings, knowledge_gaps, synthesis_depth)

        return {
            "comprehensive_summary": comprehensive_summary,
            "key_findings": key_findings,
            "knowledge_gaps": knowledge_gaps,
            "contradiction_analysis": contradiction_analysis,
            "recommendations": recommendations,
            "synthesis_quality": await self._assess_synthesis_quality(key_findings, knowledge_gaps, recommendations),
            "confidence_levels": await self._calculate_confidence_levels(key_findings, documents)
        }

    async def _generate_research_report(self, research_synthesis: Dict[str, Any], research_focus: str) -> Dict[
        str, Any]:
        """Generate structured research report"""
        return {
            "executive_summary": await self._generate_executive_summary(research_synthesis, research_focus),
            "methodology": {
                "research_approach": "Multi-source synthesis and trend analysis",
                "analysis_methods": list(self.research_methods.keys()),
                "synthesis_techniques": ["Thematic analysis", "Trend identification", "Contradiction detection"]
            },
            "key_findings": research_synthesis["key_findings"],
            "trend_analysis": await self._format_trend_analysis(research_synthesis),
            "recommendations": research_synthesis["recommendations"],
            "limitations": await self._identify_limitations(research_synthesis),
            "future_research_directions": await self._suggest_future_research(research_synthesis),
            "report_metadata": {
                "generated_date": datetime.now().isoformat(),
                "research_focus": research_focus,
                "report_version": "1.0"
            }
        }

    async def _update_knowledge_base(self, research_synthesis: Dict[str, Any], research_focus: str) -> Dict[str, Any]:
        """Update knowledge base with new research findings"""
        update_id = f"update_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.knowledge_base["summaries"][update_id] = research_synthesis["comprehensive_summary"]
        self.knowledge_base["key_findings"][update_id] = research_synthesis["key_findings"]

        # Update trend patterns
        await self._update_trend_patterns(research_synthesis)

        return {
            "update_id": update_id,
            "research_focus": research_focus,
            "elements_added": {
                "summaries": 1,
                "key_findings": len(research_synthesis["key_findings"])
            },
            "knowledge_base_metrics": {
                "total_summaries": len(self.knowledge_base["summaries"]),
                "total_findings": sum(len(findings) for findings in self.knowledge_base["key_findings"].values())
            }
        }

    # Content processing methods
    async def _extract_key_phrases(self, content: str) -> List[Dict[str, Any]]:
        """Extract key phrases from content"""
        # Simple key phrase extraction (in production, use NLP libraries)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
        word_freq = Counter(words)

        # Filter common words and get top phrases
        common_words = {'this', 'that', 'with', 'from', 'have', 'were', 'been', 'they', 'what'}
        key_phrases = []

        for word, freq in word_freq.most_common(20):
            if word not in common_words and freq >= 2:
                key_phrases.append({
                    "phrase": word,
                    "frequency": freq,
                    "significance": min(1.0, freq / 10.0)  # Normalized significance
                })

        return key_phrases[:10]  # Return top 10 key phrases

    async def _extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract entities from content"""
        # Simple entity extraction (in production, use NER)
        entities = []

        # Look for capitalized words (potential proper nouns)
        potential_entities = re.findall(r'\b[A-Z][a-z]+\b', content)
        entity_freq = Counter(potential_entities)

        for entity, freq in entity_freq.most_common(15):
            if freq >= 2:  # Only include entities mentioned multiple times
                entity_type = await self._classify_entity_type(entity, content)
                entities.append({
                    "entity": entity,
                    "type": entity_type,
                    "frequency": freq,
                    "context": await self._extract_entity_context(entity, content)
                })

        return entities

    async def _analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """Analyze sentiment of content"""
        # Simple sentiment analysis (in production, use sentiment analysis libraries)
        positive_words = {"good", "great", "excellent", "positive", "success", "benefit", "advantage"}
        negative_words = {"bad", "poor", "negative", "problem", "issue", "risk", "challenge"}

        content_lower = content.lower()
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)

        total = positive_count + negative_count
        if total == 0:
            sentiment_score = 0.5
        else:
            sentiment_score = positive_count / total

        return {
            "score": sentiment_score,
            "label": "positive" if sentiment_score > 0.6 else "negative" if sentiment_score < 0.4 else "neutral",
            "confidence": min(1.0, abs(sentiment_score - 0.5) * 2),  # Higher confidence for extreme scores
            "positive_indicators": positive_count,
            "negative_indicators": negative_count
        }

    async def _assess_readability(self, content: str) -> float:
        """Assess readability of content"""
        # Simple readability assessment
        sentences = re.split(r'[.!?]+', content)
        words = content.split()

        if len(sentences) == 0 or len(words) == 0:
            return 0.5

        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)

        # Simple readability score (higher is more readable)
        readability = 1.0 - (avg_sentence_length / 50) - (avg_word_length / 15)
        return max(0.0, min(1.0, readability))

    async def _calculate_information_density(self, content: str, key_phrases: List[Dict[str, Any]]) -> float:
        """Calculate information density of content"""
        if len(content) == 0:
            return 0.0

        # Simple information density calculation
        unique_phrases = len(key_phrases)
        total_words = len(content.split())

        density = unique_phrases / (total_words / 100)  # Phrases per 100 words
        return min(1.0, density / 10)  # Normalize to 0-1

    # Analysis methods
    async def _identify_themes(self, documents: List[Dict[str, Any]], research_focus: str) -> Dict[str, Any]:
        """Identify themes across documents"""
        all_phrases = []
        for doc in documents:
            all_phrases.extend([phrase["phrase"] for phrase in doc.get("key_phrases", [])])

        phrase_freq = Counter(all_phrases)
        common_phrases = [phrase for phrase, freq in phrase_freq.most_common(20) if freq >= 2]

        # Group related phrases into themes
        themes = await self._group_phrases_into_themes(common_phrases, research_focus)

        return {
            "identified_themes": themes,
            "theme_coverage": len(themes),
            "dominant_theme": max(themes, key=lambda x: x["strength"])["name"] if themes else "None",
            "theme_diversity": len(set(theme["name"] for theme in themes)) / len(themes) if themes else 0
        }

    async def _assess_content_quality(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess quality of research materials"""
        quality_scores = []

        for doc in documents:
            score = 0.5  # Base score

            # Adjust based on various factors
            score += doc.get("readability_score", 0) * 0.2
            score += doc.get("information_density", 0) * 0.2
            score += (1 - abs(
                doc.get("sentiment", {}).get("score", 0.5) - 0.5)) * 0.1  # Neutral sentiment preferred for research

            quality_scores.append(min(1.0, score))

        return {
            "average_quality": statistics.mean(quality_scores) if quality_scores else 0,
            "quality_distribution": {
                "excellent": len([s for s in quality_scores if s >= 0.8]),
                "good": len([s for s in quality_scores if 0.6 <= s < 0.8]),
                "fair": len([s for s in quality_scores if 0.4 <= s < 0.6]),
                "poor": len([s for s in quality_scores if s < 0.4])
            },
            "recommendations": await self._generate_quality_recommendations(quality_scores)
        }

    async def _extract_key_information(self, documents: List[Dict[str, Any]], research_focus: str) -> List[
        Dict[str, Any]]:
        """Extract key information from documents"""
        key_info = []

        for doc in documents:
            # Extract important sentences (simplified)
            sentences = re.split(r'[.!?]+', doc.get("content", ""))
            for sentence in sentences:
                if (research_focus.lower() in sentence.lower() or
                        any(phrase["phrase"] in sentence.lower() for phrase in doc.get("key_phrases", [])[:3])):
                    key_info.append({
                        "document": doc["doc_id"],
                        "information": sentence.strip(),
                        "relevance_score": await self._calculate_relevance(sentence, research_focus),
                        "source_quality": doc.get("readability_score", 0.5)
                    })

        # Sort by relevance and return top items
        key_info.sort(key=lambda x: x["relevance_score"], reverse=True)
        return key_info[:15]  # Return top 15 items

    # Additional helper methods would continue here...
    # The implementation would include all the remaining referenced methods

    async def _classify_entity_type(self, entity: str, context: str) -> str:
        """Classify entity type based on context"""
        # Simple entity classification
        entity_lower = entity.lower()
        context_lower = context.lower()

        if any(term in context_lower for term in ["company", "corporation", "inc", "ltd"]):
            return "organization"
        elif any(term in context_lower for term in ["said", "stated", "according to"]):
            return "person"
        elif any(term in entity_lower for term in ["study", "research", "analysis"]):
            return "concept"
        else:
            return "unknown"

    async def _extract_entity_context(self, entity: str, content: str) -> str:
        """Extract context around entity mention"""
        sentences = re.split(r'[.!?]+', content)
        for sentence in sentences:
            if entity in sentence:
                return sentence.strip()[:100]  # Return first 100 characters
        return ""

    async def _analyze_collection_metadata(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze metadata across document collection"""
        content_types = [doc.get("content_type", "unknown") for doc in documents]
        content_lengths = [doc.get("content_length", 0) for doc in documents]

        return {
            "content_type_distribution": dict(Counter(content_types)),
            "average_content_length": statistics.mean(content_lengths) if content_lengths else 0,
            "total_content_volume": sum(content_lengths),
            "temporal_range": await self._extract_temporal_range(documents),
            "source_variety": len(set(content_types))
        }

    async def _extract_temporal_range(self, documents: List[Dict[str, Any]]) -> Dict[str, str]:
        """Extract temporal range from documents"""
        # This would normally extract from metadata
        return {
            "earliest": datetime.now().isoformat(),
            "latest": datetime.now().isoformat(),
            "timespan_days": 0
        }

    async def _group_phrases_into_themes(self, phrases: List[str], research_focus: str) -> List[Dict[str, Any]]:
        """Group phrases into thematic categories"""
        themes = []

        # Simple thematic grouping based on research focus
        focus_related = [phrase for phrase in phrases if research_focus.lower() in phrase.lower()]
        if focus_related:
            themes.append({
                "name": f"{research_focus} Core",
                "phrases": focus_related[:5],
                "strength": len(focus_related) / len(phrases) if phrases else 0,
                "coverage": len(focus_related)
            })

        # Add general themes based on remaining phrases
        other_phrases = [phrase for phrase in phrases if phrase not in focus_related]
        if other_phrases:
            themes.append({
                "name": "Related Concepts",
                "phrases": other_phrases[:5],
                "strength": len(other_phrases) / len(phrases) if phrases else 0,
                "coverage": len(other_phrases)
            })

        return themes

    async def _calculate_relevance(self, sentence: str, research_focus: str) -> float:
        """Calculate relevance of sentence to research focus"""
        focus_terms = research_focus.lower().split()
        sentence_lower = sentence.lower()

        matches = sum(1 for term in focus_terms if term in sentence_lower)
        return matches / len(focus_terms) if focus_terms else 0.5

    async def _generate_quality_recommendations(self, quality_scores: List[float]) -> List[str]:
        """Generate recommendations for improving research quality"""
        avg_quality = statistics.mean(quality_scores) if quality_scores else 0

        if avg_quality < 0.4:
            return ["Consider adding higher quality sources", "Diversify source types"]
        elif avg_quality < 0.6:
            return ["Include more recent research materials", "Balance perspective diversity"]
        else:
            return ["Research materials are of good quality", "Continue current sourcing strategy"]
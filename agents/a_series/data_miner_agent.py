# # /agents/a_series/data_miner_agent.py
# from core.agent_base import RevenantAgentBase
# import asyncio
# from typing import Dict, List, Any, Set
# import aiohttp
# import json
# from datetime import datetime
# from urllib.parse import urlparse
# import hashlib
# import re
#
#
# class DataMinerAgent(RevenantAgentBase):
#     def __init__(self):
#         super().__init__(
#             name="DataMinerAgent",
#             description="Extracts, enriches, and transforms raw data from multiple sources into structured insights with ETL capabilities."
#         )
#         self.data_sources = {}
#         self.extraction_patterns = {}
#         self.enrichment_apis = {}
#         self.data_cache = {}
#
#     async def setup(self):
#         # Initialize data sources and extraction patterns
#         self.data_sources = {
#             "web": {
#                 "enabled": True,
#                 "rate_limit": 2,  # requests per second
#                 "timeout": 30
#             },
#             "apis": {
#                 "enabled": True,
#                 "rate_limit": 5,
#                 "timeout": 15
#             },
#             "internal_db": {
#                 "enabled": True,
#                 "rate_limit": 10,
#                 "timeout": 5
#             }
#         }
#
#         # Define extraction patterns for common data types
#         self.extraction_patterns = {
#             "emails": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
#             "urls": r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
#             "phone_numbers": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
#             "dates": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
#             "prices": r'\$\d+(?:\.\d{2})?',
#             "hashtags": r'#\w+'
#         }
#
#         # Enrichment APIs (placeholder endpoints)
#         self.enrichment_apis = {
#             "geolocation": "https://api.example.com/geolocation",
#             "sentiment": "https://api.example.com/sentiment",
#             "entity_recognition": "https://api.example.com/entities",
#             "data_validation": "https://api.example.com/validate"
#         }
#
#         # Initialize data cache
#         self.data_cache = {
#             "extracted_data": {},
#             "enriched_data": {},
#             "transformed_data": {}
#         }
#
#         await asyncio.sleep(0.1)
#
#     async def run(self, input_data: dict):
#         try:
#             query = input_data.get("query", "")
#             sources = input_data.get("sources", ["web", "apis"])
#             extraction_types = input_data.get("extraction_types", [])
#             enrichment_required = input_data.get("enrichment", True)
#
#             if not query:
#                 raise ValueError("No query provided for data mining")
#
#             # Extract data from specified sources
#             extraction_results = await self._extract_data(query, sources, extraction_types)
#
#             # Apply data enrichment if requested
#             enriched_data = {}
#             if enrichment_required:
#                 enriched_data = await self._enrich_data(extraction_results["raw_data"])
#
#             # Transform and normalize data
#             transformed_data = await self._transform_data(extraction_results["raw_data"], enriched_data)
#
#             # Perform semantic clustering and analysis
#             cluster_analysis = await self._cluster_data(transformed_data)
#
#             # Generate insights and summary
#             insights = await self._generate_insights(transformed_data, cluster_analysis)
#
#             result = {
#                 "query": query,
#                 "sources_queried": sources,
#                 "total_records": extraction_results["total_records"],
#                 "extraction_metrics": extraction_results["metrics"],
#                 "enriched_records": len(enriched_data),
#                 "clusters_identified": len(cluster_analysis["clusters"]),
#                 "data_quality_score": await self._calculate_quality_score(transformed_data),
#                 "raw_data_sample": list(extraction_results["raw_data"].values())[:3],  # Sample
#                 "enriched_data_sample": list(enriched_data.values())[:3] if enriched_data else [],
#                 "clusters": cluster_analysis["clusters"],
#                 "insights": insights,
#                 "duplicates_removed": extraction_results["duplicates_removed"],
#                 "normalization_applied": transformed_data["normalization_stats"]
#             }
#
#             return {
#                 "agent": self.name,
#                 "status": "ok",
#                 "summary": f"Data mining completed: {extraction_results['total_records']} records from {len(sources)} sources, {len(cluster_analysis['clusters'])} clusters identified",
#                 "data": result
#             }
#
#         except Exception as e:
#             return await self.on_error(e)
#
#     async def _extract_data(self, query: str, sources: List[str], extraction_types: List[str]) -> Dict[str, Any]:
#         """Extract data from multiple sources"""
#         extraction_tasks = []
#
#         for source in sources:
#             if source in self.data_sources and self.data_sources[source]["enabled"]:
#                 if source == "web":
#                     extraction_tasks.append(self._extract_from_web(query, extraction_types))
#                 elif source == "apis":
#                     extraction_tasks.append(self._extract_from_apis(query, extraction_types))
#                 elif source == "internal_db":
#                     extraction_tasks.append(self._extract_from_internal(query, extraction_types))
#
#         # Execute extractions concurrently
#         results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
#
#         # Combine and deduplicate results
#         all_data = {}
#         duplicates_removed = 0
#         total_records = 0
#
#         for result in results:
#             if isinstance(result, dict) and "data" in result:
#                 for record in result["data"]:
#                     record_hash = self._hash_record(record)
#                     if record_hash not in all_data:
#                         all_data[record_hash] = record
#                         total_records += 1
#                     else:
#                         duplicates_removed += 1
#
#         return {
#             "raw_data": all_data,
#             "total_records": total_records,
#             "duplicates_removed": duplicates_removed,
#             "metrics": {
#                 "sources_used": len(sources),
#                 "successful_extractions": len([r for r in results if not isinstance(r, Exception)]),
#                 "failed_extractions": len([r for r in results if isinstance(r, Exception)])
#             }
#         }
#
#     async def _extract_from_web(self, query: str, extraction_types: List[str]) -> Dict[str, Any]:
#         """Extract data from web sources"""
#         try:
#             # Simulate web extraction - in real implementation, this would use scraping or web APIs
#             async with aiohttp.ClientSession() as session:
#                 # This is a mock implementation
#                 mock_data = [
#                     {
#                         "id": f"web_{i}",
#                         "source": "web",
#                         "content": f"Sample web content for {query} - result {i}",
#                         "url": f"https://example.com/{query.replace(' ', '_')}_{i}",
#                         "timestamp": datetime.now().isoformat(),
#                         "confidence": 0.85 - (i * 0.1)
#                     }
#                     for i in range(5)  # Mock 5 results
#                 ]
#
#                 # Apply pattern extraction if types specified
#                 extracted_patterns = {}
#                 for record in mock_data:
#                     patterns_found = await self._extract_patterns(record["content"], extraction_types)
#                     if patterns_found:
#                         record["extracted_patterns"] = patterns_found
#                         extracted_patterns[record["id"]] = patterns_found
#
#                 return {
#                     "source": "web",
#                     "data": mock_data,
#                     "extracted_patterns": extracted_patterns,
#                     "status": "success"
#                 }
#
#         except Exception as e:
#             return {"source": "web", "data": [], "error": str(e), "status": "failed"}
#
#     async def _extract_from_apis(self, query: str, extraction_types: List[str]) -> Dict[str, Any]:
#         """Extract data from APIs"""
#         try:
#             # Simulate API extraction
#             mock_data = [
#                 {
#                     "id": f"api_{i}",
#                     "source": "api",
#                     "content": f"Structured API data for {query} - item {i}",
#                     "metadata": {
#                         "api_source": "example_api",
#                         "endpoint": f"/search?q={query}",
#                         "response_time": 120 + i * 10
#                     },
#                     "timestamp": datetime.now().isoformat(),
#                     "confidence": 0.92 - (i * 0.05)
#                 }
#                 for i in range(3)  # Mock 3 API results
#             ]
#
#             return {
#                 "source": "apis",
#                 "data": mock_data,
#                 "status": "success"
#             }
#
#         except Exception as e:
#             return {"source": "apis", "data": [], "error": str(e), "status": "failed"}
#
#     async def _extract_from_internal(self, query: str, extraction_types: List[str]) -> Dict[str, Any]:
#         """Extract data from internal databases"""
#         try:
#             # Simulate internal database query
#             mock_data = [
#                 {
#                     "id": f"internal_{i}",
#                     "source": "internal_db",
#                     "content": f"Internal database record for {query} - record {i}",
#                     "database": "revenant_analytics",
#                     "query_time": 5 + i,
#                     "timestamp": datetime.now().isoformat(),
#                     "confidence": 0.95
#                 }
#                 for i in range(2)  # Mock 2 internal records
#             ]
#
#             return {
#                 "source": "internal_db",
#                 "data": mock_data,
#                 "status": "success"
#             }
#
#         except Exception as e:
#             return {"source": "internal_db", "data": [], "error": str(e), "status": "failed"}
#
#     async def _extract_patterns(self, content: str, pattern_types: List[str]) -> Dict[str, List[str]]:
#         """Extract specific patterns from content"""
#         patterns_found = {}
#
#         for pattern_type in pattern_types:
#             if pattern_type in self.extraction_patterns:
#                 matches = re.findall(self.extraction_patterns[pattern_type], content)
#                 if matches:
#                     patterns_found[pattern_type] = list(set(matches))  # Deduplicate
#
#         return patterns_found
#
#     async def _enrich_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
#         """Enrich extracted data with additional context and metadata"""
#         enriched_data = {}
#         enrichment_tasks = []
#
#         for record_id, record in raw_data.items():
#             enrichment_tasks.append(self._enrich_single_record(record_id, record))
#
#         # Execute enrichments concurrently
#         results = await asyncio.gather(*enrichment_tasks)
#
#         for result in results:
#             if result:
#                 enriched_data[result["id"]] = result
#
#         return enriched_data
#
#     async def _enrich_single_record(self, record_id: str, record: Dict[str, Any]) -> Dict[str, Any]:
#         """Enrich a single data record"""
#         enriched_record = record.copy()
#
#         # Add semantic tags
#         enriched_record["semantic_tags"] = await self._generate_semantic_tags(record)
#
#         # Add sentiment analysis
#         if "content" in record:
#             enriched_record["sentiment"] = await self._analyze_sentiment(record["content"])
#
#         # Add entity recognition
#         enriched_record["entities"] = await self._extract_entities(record)
#
#         # Add data quality score
#         enriched_record["data_quality"] = await self._assess_data_quality(record)
#
#         # Add enrichment timestamp
#         enriched_record["enriched_at"] = datetime.now().isoformat()
#
#         return enriched_record
#
#     async def _transform_data(self, raw_data: Dict[str, Any], enriched_data: Dict[str, Any]) -> Dict[str, Any]:
#         """Transform and normalize data into consistent format"""
#         transformed_records = {}
#         normalization_stats = {
#             "fields_normalized": 0,
#             "records_standardized": 0,
#             "formats_converted": 0
#         }
#
#         all_data = {**raw_data, **enriched_data}
#
#         for record_id, record in all_data.items():
#             transformed_record = await self._normalize_record(record)
#             transformed_records[record_id] = transformed_record
#
#             # Update normalization stats
#             if transformed_record != record:
#                 normalization_stats["records_standardized"] += 1
#
#         return {
#             "transformed_data": transformed_records,
#             "normalization_stats": normalization_stats,
#             "total_transformed": len(transformed_records)
#         }
#
#     async def _normalize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
#         """Normalize a single record to standard format"""
#         normalized = record.copy()
#
#         # Standardize timestamp format
#         if "timestamp" in normalized:
#             try:
#                 # Convert to ISO format if needed
#                 if isinstance(normalized["timestamp"], str) and "T" not in normalized["timestamp"]:
#                     # Simple date normalization
#                     normalized["timestamp"] = datetime.now().isoformat()
#             except:
#                 normalized["timestamp"] = datetime.now().isoformat()
#
#         # Normalize confidence scores to 0-1 range
#         if "confidence" in normalized:
#             confidence = normalized["confidence"]
#             if isinstance(confidence, (int, float)) and confidence > 1:
#                 normalized["confidence"] = confidence / 100.0
#
#         # Standardize field names
#         field_mapping = {
#             "url": "source_url",
#             "link": "source_url",
#             "score": "confidence",
#             "value": "content"
#         }
#
#         for old_field, new_field in field_mapping.items():
#             if old_field in normalized and new_field not in normalized:
#                 normalized[new_field] = normalized.pop(old_field)
#
#         return normalized
#
#     async def _cluster_data(self, transformed_data: Dict[str, Any]) -> Dict[str, Any]:
#         """Perform semantic clustering on the data"""
#         records = transformed_data.get("transformed_data", {})
#
#         if not records:
#             return {"clusters": [], "cluster_metrics": {}}
#
#         # Simple content-based clustering (in production, use ML clustering)
#         clusters = {}
#
#         for record_id, record in records.items():
#             content = record.get("content", "").lower()
#
#             # Simple keyword-based clustering
#             cluster_key = "other"
#             if any(word in content for word in ["python", "code", "programming"]):
#                 cluster_key = "technical"
#             elif any(word in content for word in ["write", "content", "article"]):
#                 cluster_key = "content_creation"
#             elif any(word in content for word in ["search", "research", "find"]):
#                 cluster_key = "research"
#             elif any(word in content for word in ["image", "design", "visual"]):
#                 cluster_key = "creative"
#
#             if cluster_key not in clusters:
#                 clusters[cluster_key] = []
#
#             clusters[cluster_key].append({
#                 "record_id": record_id,
#                 "content_preview": content[:100] + "..." if len(content) > 100 else content,
#                 "confidence": record.get("confidence", 0),
#                 "source": record.get("source", "unknown")
#             })
#
#         return {
#             "clusters": clusters,
#             "cluster_metrics": {
#                 "total_clusters": len(clusters),
#                 "largest_cluster": max(len(cluster) for cluster in clusters.values()) if clusters else 0,
#                 "average_cluster_size": len(records) / len(clusters) if clusters else 0
#             }
#         }
#
#     async def _generate_insights(self, transformed_data: Dict[str, Any], cluster_analysis: Dict[str, Any]) -> List[str]:
#         """Generate insights from the mined data"""
#         insights = []
#         records = transformed_data.get("transformed_data", {})
#         clusters = cluster_analysis.get("clusters", {})
#
#         if not records:
#             return ["No data available for insight generation"]
#
#         # Basic insights based on data characteristics
#         total_records = len(records)
#         sources_used = set(record.get("source", "unknown") for record in records.values())
#         avg_confidence = sum(record.get("confidence", 0) for record in records.values()) / total_records
#
#         insights.append(f"Processed {total_records} records from {len(sources_used)} sources")
#         insights.append(f"Average data confidence: {avg_confidence:.2f}")
#
#         # Cluster-based insights
#         if clusters:
#             largest_cluster = max(clusters.items(), key=lambda x: len(x[1]))
#             insights.append(f"Largest topic cluster: '{largest_cluster[0]}' with {len(largest_cluster[1])} records")
#             insights.append(f"Identified {len(clusters)} distinct topic areas")
#
#         # Data quality insights
#         low_confidence_records = sum(1 for record in records.values() if record.get("confidence", 0) < 0.7)
#         if low_confidence_records > 0:
#             insights.append(f"{low_confidence_records} records have low confidence and may need verification")
#
#         return insights
#
#     async def _generate_semantic_tags(self, record: Dict[str, Any]) -> List[str]:
#         """Generate semantic tags for a record"""
#         content = record.get("content", "").lower()
#         tags = []
#
#         # Simple keyword-based tagging
#         keyword_categories = {
#             "technical": ["python", "code", "api", "database", "server"],
#             "creative": ["design", "create", "write", "content", "image"],
#             "analytical": ["analyze", "research", "data", "statistics", "trend"],
#             "commercial": ["product", "price", "buy", "sell", "market"]
#         }
#
#         for category, keywords in keyword_categories.items():
#             if any(keyword in content for keyword in keywords):
#                 tags.append(category)
#
#         return tags
#
#     async def _analyze_sentiment(self, content: str) -> float:
#         """Analyze sentiment of content (simplified)"""
#         positive_words = ["good", "great", "excellent", "amazing", "positive", "success"]
#         negative_words = ["bad", "poor", "terrible", "negative", "failure", "problem"]
#
#         content_lower = content.lower()
#         positive_count = sum(1 for word in positive_words if word in content_lower)
#         negative_count = sum(1 for word in negative_words if word in content_lower)
#
#         total = positive_count + negative_count
#         if total == 0:
#             return 0.5  # Neutral
#
#         return positive_count / total
#
#     async def _extract_entities(self, record: Dict[str, Any]) -> List[str]:
#         """Extract entities from record content (simplified)"""
#         content = record.get("content", "")
#         # Simple entity extraction - in production, use NER models
#         entities = []
#
#         # Extract potential proper nouns (capitalized words)
#         words = content.split()
#         potential_entities = [word for word in words if word.istitle() and len(word) > 2]
#
#         return list(set(potential_entities))[:5]  # Return top 5 unique entities
#
#     async def _assess_data_quality(self, record: Dict[str, Any]) -> float:
#         """Assess quality of a data record"""
#         quality_score = 0.5  # Base score
#
#         # Check for completeness
#         required_fields = ["content", "source", "timestamp"]
#         present_fields = sum(1 for field in required_fields if field in record)
#         completeness = present_fields / len(required_fields)
#         quality_score += completeness * 0.3
#
#         # Check confidence
#         confidence = record.get("confidence", 0)
#         quality_score += confidence * 0.2
#
#         return min(1.0, quality_score)
#
#     async def _calculate_quality_score(self, transformed_data: Dict[str, Any]) -> float:
#         """Calculate overall data quality score"""
#         records = transformed_data.get("transformed_data", {})
#         if not records:
#             return 0.0
#
#         total_quality = sum(record.get("data_quality", 0.5) for record in records.values())
#         return total_quality / len(records)
#
#     def _hash_record(self, record: Dict[str, Any]) -> str:
#         """Generate hash for record deduplication"""
#         record_str = json.dumps(record, sort_keys=True)
#         return hashlib.md5(record_str.encode()).hexdigest()


# /agents/a_series/data_miner_agent.py
from core.agent_base import RevenantAgentBase
import asyncio
from typing import Dict, List, Any
from datetime import datetime


# TODO: review - aiohttp import added to requirements.txt for async HTTP
try:
    import aiohttp
except ImportError:
    aiohttp = None


class DataMinerAgent(RevenantAgentBase):
    """
    Extracts, enriches, and transforms raw data from multiple sources into structured insights.

    Input:
        - query (str): Search query for data mining
        - sources (list): Data sources to query (e.g., ["web", "apis", "internal_db"])
        - extraction_types (list): Types of data patterns to extract
        - enrichment (bool): Whether to enrich extracted data

    Output:
        - total_records (int): Number of records extracted
        - enriched_records (int): Number of enriched records
        - clusters_identified (int): Number of data clusters found
        - data_quality_score (float): Overall data quality (0-1)
        - insights (list): Generated insights from mined data
    """

    metadata = {
        "name": "DataMinerAgent",
        "series": "a_series",
        "version": "0.1.0",
        "description": "Extracts, enriches, and transforms raw data from multiple sources into structured insights with ETL capabilities."
    }

    def __init__(self):
        super().__init__(
            name="DataMinerAgent",
            description="Extracts, enriches, and transforms raw data from multiple sources into structured insights with ETL capabilities."
        )
        self.data_sources = {}
        self.extraction_patterns = {}
        self.enrichment_apis = {}
        self.data_cache = {}
        self.metadata = DataMinerAgent.metadata

    async def setup(self):
        # Initialize data sources and extraction patterns
        self.data_sources = {
            "web": {
                "enabled": True,
                "rate_limit": 2,  # requests per second
                "timeout": 30
            },
            "apis": {
                "enabled": True,
                "rate_limit": 5,
                "timeout": 15
            },
            "internal_db": {
                "enabled": True,
                "rate_limit": 10,
                "timeout": 5
            }
        }

        # Define extraction patterns for common data types
        self.extraction_patterns = {
            "emails": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "urls": r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            "phone_numbers": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "dates": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            "prices": r'\$\d+(?:\.\d{2})?',
            "hashtags": r'#\w+'
        }

        # Enrichment APIs (placeholder endpoints)
        self.enrichment_apis = {
            "geolocation": "https://api.example.com/geolocation",
            "sentiment": "https://api.example.com/sentiment",
            "entity_recognition": "https://api.example.com/entities",
            "data_validation": "https://api.example.com/validate"
        }

        # Initialize data cache
        self.data_cache = {
            "extracted_data": {},
            "enriched_data": {},
            "transformed_data": {}
        }

        await asyncio.sleep(0.1)

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            query = input_data.get("query", "")
            sources = input_data.get("sources", ["web", "apis"])
            extraction_types = input_data.get("extraction_types", [])
            enrichment_required = input_data.get("enrichment", True)

            if not query:
                raise ValueError("No query provided for data mining")

            # Extract data from specified sources
            extraction_results = await self._extract_data(query, sources, extraction_types)

            # Apply data enrichment if requested
            enriched_data = {}
            if enrichment_required:
                enriched_data = await self._enrich_data(extraction_results["raw_data"])

            # Transform and normalize data
            transformed_data = await self._transform_data(extraction_results["raw_data"], enriched_data)

            # Perform semantic clustering and analysis
            cluster_analysis = await self._cluster_data(transformed_data)

            # Generate insights and summary
            insights = await self._generate_insights(transformed_data, cluster_analysis)

            result = {
                "query": query,
                "sources_queried": sources,
                "total_records": extraction_results["total_records"],
                "extraction_metrics": extraction_results["metrics"],
                "enriched_records": len(enriched_data),
                "clusters_identified": len(cluster_analysis["clusters"]),
                "data_quality_score": await self._calculate_quality_score(transformed_data),
                "raw_data_sample": list(extraction_results["raw_data"].values())[:3],  # Sample
                "enriched_data_sample": list(enriched_data.values())[:3] if enriched_data else [],
                "clusters": cluster_analysis["clusters"],
                "insights": insights,
                "duplicates_removed": extraction_results["duplicates_removed"],
                "normalization_applied": transformed_data["normalization_stats"]
            }

            return {
                "agent": self.name,
                "status": "ok",
                "summary": f"Data mining completed: {extraction_results['total_records']} records from {len(sources)} sources, {len(cluster_analysis['clusters'])} clusters identified",
                "data": result
            }

        except Exception as e:
            return await self.on_error(e)

    async def _extract_from_web(self, query: str, extraction_types: List[str]) -> Dict[str, Any]:
        """Extract data from web sources"""
        global mock_data
        try:
            # TODO: review - aiohttp is optional; using mock data if not available
            if aiohttp is None:
                # Use mock data if aiohttp not installed
                mock_data = [
                    {
                        "id": f"web_{i}",
                        "source": "web",
                        "content": f"Sample web content for {query} - result {i}",
                        "url": f"https://example.com/{query.replace(' ', '_')}_{i}",
                        "timestamp": datetime.now().isoformat(),
                        "confidence": 0.85 - (i * 0.1)
                    }
                    for i in range(5)  # Mock 5 results
                ]
            else:
                # Simulate web extraction - in real implementation, this would use scraping or web APIs
                async with aiohttp.ClientSession() as session:
                    # This is a mock implementation
                    mock_data = [
                        {
                            "id": f"web_{i}",
                            "source": "web",
                            "content": f"Sample web content for {query} - result {i}",
                            "url": f"https://example.com/{query.replace(' ', '_')}_{i}",
                            "timestamp": datetime.now().isoformat(),
                            "confidence": 0.85 - (i * 0.1)
                        }
                        for i in range(5)  # Mock 5 results
                    ]

            # Apply pattern extraction if types specified
            extracted_patterns = {}
            for record in mock_data:
                patterns_found = await self._extract_patterns(record["content"], extraction_types)
                if patterns_found:
                    record["extracted_patterns"] = patterns_found
                    extracted_patterns[record["id"]] = patterns_found

            return {
                "source": "web",
                "data": mock_data,
                "extracted_patterns": extracted_patterns,
                "status": "success"
            }

        except Exception as e:
            return {"source": "web", "data": [], "error": str(e), "status": "failed"}

# ... existing code ...
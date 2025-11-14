# /agents/c_series/knowledge_graph_agent.py
from core.agent_base import RevenantAgentBase
import asyncio
from typing import Dict, List, Any
from datetime import datetime
import hashlib
import networkx as nx
from collections import defaultdict



class KnowledgeGraphAgent(RevenantAgentBase):
    """Build and maintain semantic knowledge graphs from agent outputs with entity extraction and cross-agent knowledge linking."""
    metadata = {
        "name": "KnowledgeGraphAgent",
        "version": "1.0.0",
        "series": "c_series",
        "description": "Constructs and maintains dynamic knowledge graphs from multi-agent outputs with semantic reasoning and relationship mapping",
        "module": "agents.c_series.knowledge_graph_agent"

    }
    def __init__(self):
        super().__init__(
            name=self.metadata['name'],
            description=self.metadata['description'])
        self.knowledge_graph = nx.MultiDiGraph()
        self.entity_registry = {}
        self.relationship_patterns = {}

    async def setup(self):
        # Initialize knowledge graph and entity registry
        self.knowledge_graph = nx.MultiDiGraph()
        self.entity_registry = {
            "agents": {},
            "concepts": {},
            "relationships": {},
            "timestamps": {}
        }

        # Define relationship patterns for semantic reasoning
        self.relationship_patterns = {
            "hierarchical": ["is_a", "part_of", "contains", "belongs_to"],
            "causal": ["causes", "affects", "influences", "leads_to"],
            "temporal": ["before", "after", "during", "precedes"],
            "associative": ["related_to", "associated_with", "similar_to", "connected_to"],
            "functional": ["uses", "produces", "consumes", "generates"]
        }

        await asyncio.sleep(0.1)

    async def run(self, input_data: dict):
        try:
            # Validate input
            agent_outputs = input_data.get("agent_outputs", [])
            operation = input_data.get("operation", "update")
            graph_depth = input_data.get("depth", 2)

            if not agent_outputs:
                raise ValueError("No agent outputs provided for knowledge graph construction")

            # Extract entities and relationships from agent outputs
            extraction_results = await self._extract_knowledge_components(agent_outputs)

            # Update knowledge graph based on operation
            if operation == "update":
                graph_updates = await self._update_knowledge_graph(extraction_results)
            elif operation == "query":
                graph_updates = await self._query_knowledge_graph(extraction_results)
            elif operation == "reason":
                graph_updates = await self._perform_graph_reasoning(extraction_results, graph_depth)
            else:
                graph_updates = await self._update_knowledge_graph(extraction_results)

            # Generate insights from graph structure
            graph_insights = await self._generate_graph_insights()

            # Identify cross-agent knowledge patterns
            cross_agent_patterns = await self._identify_cross_agent_patterns(agent_outputs)

            result = {
                "graph_metadata": {
                    "total_nodes": len(self.knowledge_graph.nodes()),
                    "total_edges": len(self.knowledge_graph.edges()),
                    "graph_density": nx.density(self.knowledge_graph),
                    "connected_components": nx.number_weakly_connected_components(self.knowledge_graph)
                },
                "extraction_results": extraction_results,
                "graph_updates": graph_updates,
                "graph_insights": graph_insights,
                "cross_agent_patterns": cross_agent_patterns,
                "semantic_clusters": await self._identify_semantic_clusters(),
                "knowledge_gaps": await self._identify_knowledge_gaps(),
                "reasoning_paths": await self._generate_reasoning_paths(graph_depth)
            }

            return {
                "agent": self.name,
                "status": "ok",
                "summary": f"Knowledge graph updated: {graph_updates['entities_added']} new entities, {graph_updates['relationships_added']} new relationships. Graph now has {len(self.knowledge_graph.nodes())} nodes and {len(self.knowledge_graph.edges())} edges.",
                "data": result
            }

        except Exception as e:
            return await self.on_error(str(e))

    async def _extract_knowledge_components(self, agent_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract entities and relationships from agent outputs"""
        entities = []
        relationships = []
        agent_contexts = []

        for output in agent_outputs:
            agent_name = output.get("agent", "unknown")
            data = output.get("data", {})

            # Extract entities from agent data
            agent_entities = await self._extract_entities_from_data(data, agent_name)
            entities.extend(agent_entities)

            # Extract relationships
            agent_relationships = await self._extract_relationships(agent_entities, data, agent_name)
            relationships.extend(agent_relationships)

            # Store agent context
            agent_contexts.append({
                "agent": agent_name,
                "timestamp": output.get("timestamp", datetime.now().isoformat()),
                "entities_extracted": len(agent_entities),
                "relationships_extracted": len(agent_relationships)
            })

        return {
            "entities": entities,
            "relationships": relationships,
            "agent_contexts": agent_contexts,
            "extraction_metrics": {
                "total_entities": len(entities),
                "total_relationships": len(relationships),
                "unique_entities": len(set(e["id"] for e in entities)),
                "agents_processed": len(agent_outputs)
            }
        }

    async def _extract_entities_from_data(self, data: Dict[str, Any], source_agent: str) -> List[Dict[str, Any]]:
        """Extract entities from structured data"""
        entities = []

        async def extract_from_value(value, path="", depth=0):
            if depth > 3:  # Prevent infinite recursion
                return

            if isinstance(value, dict):
                for key, val in value.items():
                    current_path = f"{path}.{key}" if path else key

                    # Consider keys as potential entities
                    if self._is_entity_candidate(key, val):
                        entity_id = self._generate_entity_id(key, source_agent)
                        entities.append({
                            "id": entity_id,
                            "name": key,
                            "type": await self._classify_entity_type(key, val),
                            "value": str(val)[:100] if val else None,  # Truncate long values
                            "source_agent": source_agent,
                            "properties": await self._extract_entity_properties(val),
                            "confidence": await self._calculate_entity_confidence(key, val),
                            "extraction_path": current_path
                        })

                    await extract_from_value(val, current_path, depth + 1)

            elif isinstance(value, list):
                for i, item in enumerate(value):
                    await extract_from_value(item, f"{path}[{i}]", depth + 1)

        await extract_from_value(data)
        return entities

    async def _extract_relationships(self, entities: List[Dict[str, Any]], data: Dict[str, Any], source_agent: str) -> \
    List[Dict[str, Any]]:
        """Extract relationships between entities"""
        relationships = []

        if len(entities) < 2:
            return relationships

        # Create relationships based on entity co-occurrence and data structure
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i + 1:], i + 1):
                if await self._should_create_relationship(entity1, entity2, data):
                    relationship = await self._create_relationship(entity1, entity2, data, source_agent)
                    if relationship:
                        relationships.append(relationship)

        return relationships

    async def _update_knowledge_graph(self, extraction_results: Dict[str, Any]) -> Dict[str, Any]:
        """Update knowledge graph with new entities and relationships"""
        entities_added = 0
        relationships_added = 0
        entities_updated = 0

        # Add entities to graph
        for entity in extraction_results["entities"]:
            entity_id = entity["id"]

            if not self.knowledge_graph.has_node(entity_id):
                self.knowledge_graph.add_node(entity_id, **entity)
                entities_added += 1
            else:
                # Update existing entity
                current_data = self.knowledge_graph.nodes[entity_id]
                updated_data = {**current_data, **entity}
                self.knowledge_graph.nodes[entity_id].update(updated_data)
                entities_updated += 1

            # Update entity registry
            self.entity_registry["agents"][entity_id] = entity.get("source_agent", "unknown")
            self.entity_registry["timestamps"][entity_id] = datetime.now().isoformat()

        # Add relationships to graph
        for relationship in extraction_results["relationships"]:
            source = relationship["source"]
            target = relationship["target"]
            relationship_type = relationship["type"]

            if (self.knowledge_graph.has_node(source) and
                    self.knowledge_graph.has_node(target)):

                # Check if relationship already exists
                existing_edges = self.knowledge_graph.get_edge_data(source, target)
                relationship_exists = False

                if existing_edges:
                    for key, edge_data in existing_edges.items():
                        if edge_data.get("type") == relationship_type:
                            relationship_exists = True
                            break

                if not relationship_exists:
                    self.knowledge_graph.add_edge(source, target, **relationship)
                    relationships_added += 1

        return {
            "entities_added": entities_added,
            "entities_updated": entities_updated,
            "relationships_added": relationships_added,
            "timestamp": datetime.now().isoformat()
        }

    async def _query_knowledge_graph(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Query the knowledge graph for specific information"""
        query_type = query_params.get("query_type", "entity_search")
        results = {}

        if query_type == "entity_search":
            search_term = query_params.get("search_term", "")
            results = await self._search_entities(search_term)

        elif query_type == "relationship_path":
            source = query_params.get("source")
            target = query_params.get("target")
            results = await self._find_relationship_paths(source, target)

        elif query_type == "subgraph_extraction":
            central_entity = query_params.get("central_entity")
            radius = query_params.get("radius", 2)
            results = await self._extract_subgraph(central_entity, radius)

        return {
            "query_type": query_type,
            "results": results,
            "execution_time": datetime.now().isoformat()
        }

    async def _perform_graph_reasoning(self, reasoning_params: Dict[str, Any], depth: int) -> Dict[str, Any]:
        """Perform reasoning on the knowledge graph"""
        reasoning_type = reasoning_params.get("reasoning_type", "inference")
        reasoning_results = {}

        if reasoning_type == "inference":
            reasoning_results = await self._perform_logical_inference(depth)

        elif reasoning_type == "pattern_matching":
            reasoning_results = await self._find_patterns_in_graph()

        elif reasoning_type == "hypothesis_testing":
            hypothesis = reasoning_params.get("hypothesis")
            reasoning_results = await self._test_hypothesis(hypothesis, depth)

        return {
            "reasoning_type": reasoning_type,
            "results": reasoning_results,
            "reasoning_depth": depth
        }

    async def _generate_graph_insights(self) -> Dict[str, Any]:
        """Generate insights from graph structure and content"""
        if len(self.knowledge_graph.nodes()) == 0:
            return {"message": "Insufficient data for insights"}

        # Calculate graph metrics
        centrality = nx.degree_centrality(self.knowledge_graph)
        betweenness = nx.betweenness_centrality(self.knowledge_graph)

        # Find key entities
        top_central_entities = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        top_betweenness_entities = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "key_entities": {
                "most_connected": [{"entity": ent, "score": score} for ent, score in top_central_entities],
                "most_influential": [{"entity": ent, "score": score} for ent, score in top_betweenness_entities]
            },
            "graph_structure": {
                "average_degree": sum(dict(self.knowledge_graph.degree()).values()) / len(self.knowledge_graph.nodes()),
                "clustering_coefficient": nx.average_clustering(self.knowledge_graph.to_undirected()),
                "assortativity": nx.degree_assortativity_coefficient(self.knowledge_graph)
            },
            "domain_coverage": await self._analyze_domain_coverage(),
            "knowledge_completeness": await self._assess_knowledge_completeness()
        }

    async def _identify_cross_agent_patterns(self, agent_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify patterns across different agent outputs"""
        agent_contributions = defaultdict(list)

        for output in agent_outputs:
            agent_name = output.get("agent", "unknown")
            data = output.get("data", {})

            # Extract key contributions from each agent
            contribution = await self._extract_agent_contribution(data, agent_name)
            agent_contributions[agent_name].append(contribution)

        # Analyze patterns across agents
        patterns = await self._analyze_cross_agent_patterns(agent_contributions)

        return {
            "agent_contributions": dict(agent_contributions),
            "cross_agent_patterns": patterns,
            "collaboration_opportunities": await self._identify_collaboration_opportunities(agent_contributions)
        }

    async def _identify_semantic_clusters(self) -> Dict[str, Any]:
        """Identify semantic clusters in the knowledge graph"""
        if len(self.knowledge_graph.nodes()) < 3:
            return {"clusters": [], "metrics": {}}

        try:
            # Convert to undirected graph for community detection
            undirected_graph = self.knowledge_graph.to_undirected()

            # Use community detection algorithms
            communities = nx.algorithms.community.greedy_modularity_communities(undirected_graph)

            clusters = []
            for i, community in enumerate(communities):
                cluster_entities = [self.knowledge_graph.nodes[node] for node in community]
                cluster_theme = await self._identify_cluster_theme(cluster_entities)

                clusters.append({
                    "cluster_id": i,
                    "theme": cluster_theme,
                    "size": len(community),
                    "entities": cluster_entities[:10]  # Limit to first 10 entities
                })

            return {
                "clusters": clusters,
                "metrics": {
                    "total_clusters": len(clusters),
                    "modularity": nx.algorithms.community.modularity(undirected_graph, communities),
                    "coverage": sum(len(c) for c in communities) / len(self.knowledge_graph.nodes())
                }
            }
        except:
            return {"clusters": [], "metrics": {"error": "Community detection failed"}}

    async def _identify_knowledge_gaps(self) -> List[Dict[str, Any]]:
        """Identify gaps in the knowledge graph"""
        gaps = []

        # Find isolated entities
        isolated_nodes = list(nx.isolates(self.knowledge_graph))
        if isolated_nodes:
            gaps.append({
                "type": "isolated_entities",
                "description": f"{len(isolated_nodes)} entities have no relationships",
                "entities": isolated_nodes[:5]
            })

        # Find entities with low connectivity
        if len(self.knowledge_graph.nodes()) > 10:
            degrees = dict(self.knowledge_graph.degree())
            low_connectivity = [node for node, degree in degrees.items() if degree <= 1]
            if low_connectivity:
                gaps.append({
                    "type": "low_connectivity",
                    "description": f"{len(low_connectivity)} entities have minimal connections",
                    "entities": low_connectivity[:5]
                })

        return gaps

    async def _generate_reasoning_paths(self, max_depth: int) -> List[Dict[str, Any]]:
        """Generate potential reasoning paths through the knowledge graph"""
        if len(self.knowledge_graph.nodes()) < 3:
            return []

        reasoning_paths = []

        # Find central entities as starting points
        centrality = nx.degree_centrality(self.knowledge_graph)
        central_entities = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]

        for entity, _ in central_entities:
            paths_from_entity = await self._explore_reasoning_paths(entity, max_depth)
            reasoning_paths.extend(paths_from_entity)

        return reasoning_paths[:5]  # Return top 5 reasoning paths

    # Helper methods
    def _is_entity_candidate(self, key: str, value: Any) -> bool:
        """Determine if a key-value pair represents a potential entity"""
        # Exclude common non-entity keys
        excluded_keys = {"timestamp", "status", "id", "summary", "metadata"}

        if key.lower() in excluded_keys:
            return False

        # Consider strings of reasonable length and certain patterns as entities
        if isinstance(value, str) and 2 <= len(value) <= 100:
            return True

        # Consider numbers with context as entities
        if isinstance(value, (int, float)) and key not in ["count", "total", "score"]:
            return True

        # Consider dictionaries with substantial content
        if isinstance(value, dict) and len(value) > 0:
            return True

        return False

    async def _classify_entity_type(self, key: str, value: Any) -> str:
        """Classify the type of entity"""
        key_lower = key.lower()

        if any(term in key_lower for term in ["agent", "system", "module"]):
            return "agent"
        elif any(term in key_lower for term in ["user", "person", "customer"]):
            return "person"
        elif any(term in key_lower for term in ["data", "information", "knowledge"]):
            return "concept"
        elif any(term in key_lower for term in ["task", "job", "workflow"]):
            return "task"
        elif any(term in key_lower for term in ["tool", "resource", "asset"]):
            return "resource"
        elif isinstance(value, (int, float)):
            return "metric"
        else:
            return "concept"

    def _generate_entity_id(self, name: str, source: str) -> str:
        """Generate unique entity ID"""
        unique_string = f"{name}_{source}_{datetime.now().timestamp()}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:12]

    async def _extract_entity_properties(self, value: Any) -> Dict[str, Any]:
        """Extract properties from entity value"""
        properties = {}

        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, (str, int, float, bool)):
                    properties[k] = v
        elif isinstance(value, (str, int, float, bool)):
            properties["value"] = value

        return properties

    async def _calculate_entity_confidence(self, key: str, value: Any) -> float:
        """Calculate confidence score for entity extraction"""
        confidence = 0.5  # Base confidence

        # Increase confidence for meaningful keys
        if len(key) >= 3 and not key.isnumeric():
            confidence += 0.2

        # Increase confidence for substantial values
        if isinstance(value, dict) and len(value) > 0:
            confidence += 0.2
        elif isinstance(value, str) and len(value) >= 5:
            confidence += 0.1

        return min(1.0, confidence)

    async def _should_create_relationship(self, entity1: Dict[str, Any], entity2: Dict[str, Any],
                                          data: Dict[str, Any]) -> bool:
        """Determine if a relationship should be created between two entities"""
        # Entities from same agent and extraction path are likely related
        if (entity1["source_agent"] == entity2["source_agent"] and
                entity1["extraction_path"] and entity2["extraction_path"] and
                entity1["extraction_path"].split('.')[0] == entity2["extraction_path"].split('.')[0]):
            return True

        # Entities with similar types might be related
        if entity1["type"] == entity2["type"]:
            return True

        return False

    async def _create_relationship(self, entity1: Dict[str, Any], entity2: Dict[str, Any], data: Dict[str, Any],
                                   source_agent: str) -> Dict[str, Any]:
        """Create a relationship between two entities"""
        relationship_types = await self._determine_relationship_type(entity1, entity2, data)

        return {
            "source": entity1["id"],
            "target": entity2["id"],
            "type": relationship_types[0],  # Primary relationship type
            "alternative_types": relationship_types[1:],
            "source_agent": source_agent,
            "confidence": await self._calculate_relationship_confidence(entity1, entity2),
            "timestamp": datetime.now().isoformat(),
            "properties": {
                "source_type": entity1["type"],
                "target_type": entity2["type"],
                "extraction_context": f"From {source_agent} data"
            }
        }

    async def _determine_relationship_type(self, entity1: Dict[str, Any], entity2: Dict[str, Any],
                                           data: Dict[str, Any]) -> List[str]:
        """Determine the type of relationship between entities"""
        relationships = []

        # Hierarchical relationships
        if (entity1["type"] == "agent" and entity2["type"] == "task") or \
                (entity1["type"] == "task" and entity2["type"] == "resource"):
            relationships.append("uses")

        # Causal relationships based on entity names
        entity1_name = entity1["name"].lower()
        entity2_name = entity2["name"].lower()

        if any(term in entity1_name for term in ["cause", "create", "generate"]) and \
                any(term in entity2_name for term in ["effect", "result", "output"]):
            relationships.append("causes")

        # Default to associative relationship
        if not relationships:
            relationships.append("related_to")

        return relationships

    async def _calculate_relationship_confidence(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> float:
        """Calculate confidence score for relationship"""
        confidence = 0.5

        # Increase confidence if entities have high individual confidence
        confidence += (entity1["confidence"] + entity2["confidence"]) * 0.2

        # Increase confidence for same source agent
        if entity1["source_agent"] == entity2["source_agent"]:
            confidence += 0.1

        return min(1.0, confidence)

    async def _search_entities(self, search_term: str) -> Dict[str, Any]:
        """Search for entities in the knowledge graph"""
        matching_entities = []

        for node_id, node_data in self.knowledge_graph.nodes(data=True):
            if (search_term.lower() in node_data.get("name", "").lower() or
                    search_term.lower() in str(node_data.get("value", "")).lower()):

                # Get relationships for this entity
                relationships = []
                for neighbor in self.knowledge_graph.neighbors(node_id):
                    edge_data = self.knowledge_graph.get_edge_data(node_id, neighbor)
                    if edge_data:
                        for key, data in edge_data.items():
                            relationships.append({
                                "target": neighbor,
                                "type": data.get("type", "unknown"),
                                "confidence": data.get("confidence", 0.5)
                            })

                matching_entities.append({
                    "entity": node_data,
                    "relationships": relationships[:5]  # Limit to 5 relationships
                })

        return {
            "search_term": search_term,
            "matches_found": len(matching_entities),
            "entities": matching_entities[:10]  # Limit to 10 entities
        }

    async def _find_relationship_paths(self, source: str, target: str) -> Dict[str, Any]:
        """Find relationship paths between two entities"""
        if not self.knowledge_graph.has_node(source) or not self.knowledge_graph.has_node(target):
            return {"error": "Source or target entity not found"}

        try:
            paths = list(nx.all_simple_paths(self.knowledge_graph, source, target, cutoff=3))

            path_details = []
            for path in paths[:5]:  # Limit to 5 paths
                path_info = await self._describe_path(path)
                path_details.append(path_info)

            return {
                "source": source,
                "target": target,
                "paths_found": len(paths),
                "shortest_path_length": len(min(paths, key=len)) if paths else 0,
                "path_details": path_details
            }
        except:
            return {"error": "Path finding failed"}

    async def _extract_subgraph(self, central_entity: str, radius: int) -> Dict[str, Any]:
        """Extract a subgraph around a central entity"""
        if not self.knowledge_graph.has_node(central_entity):
            return {"error": "Central entity not found"}

        try:
            # Get entities within specified radius
            neighbors = nx.single_source_shortest_path_length(self.knowledge_graph, central_entity, radius)
            subgraph_nodes = list(neighbors.keys())
            subgraph = self.knowledge_graph.subgraph(subgraph_nodes)

            return {
                "central_entity": central_entity,
                "radius": radius,
                "nodes_in_subgraph": len(subgraph_nodes),
                "edges_in_subgraph": len(subgraph.edges()),
                "subgraph_density": nx.density(subgraph)
            }
        except:
            return {"error": "Subgraph extraction failed"}

    async def _perform_logical_inference(self, depth: int) -> Dict[str, Any]:
        """Perform logical inference on the knowledge graph"""
        inferences = []

        # Simple inference: transitive relationships
        for node in list(self.knowledge_graph.nodes())[:10]:  # Limit for performance
            node_inferences = await self._infer_transitive_relationships(node, depth)
            inferences.extend(node_inferences)

        return {
            "inferences_made": len(inferences),
            "inference_depth": depth,
            "inferences": inferences[:10]  # Limit output
        }

    async def _find_patterns_in_graph(self) -> Dict[str, Any]:
        """Find patterns in the knowledge graph"""
        patterns = []

        # Find common relationship patterns
        relationship_counts = defaultdict(int)
        for u, v, data in self.knowledge_graph.edges(data=True):
            relationship_type = data.get("type", "unknown")
            relationship_counts[relationship_type] += 1

        common_relationships = sorted(relationship_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        for rel_type, count in common_relationships:
            patterns.append({
                "type": "common_relationship",
                "pattern": f"Relationship '{rel_type}' appears {count} times",
                "confidence": min(1.0, count / len(self.knowledge_graph.edges()))
            })

        return {
            "patterns_identified": len(patterns),
            "patterns": patterns
        }

    async def _test_hypothesis(self, hypothesis: str, depth: int) -> Dict[str, Any]:
        """Test a hypothesis against the knowledge graph"""
        # This is a simplified hypothesis testing
        # In a real implementation, this would use more sophisticated logic

        return {
            "hypothesis": hypothesis,
            "supported": await self._check_hypothesis_support(hypothesis),
            "confidence": 0.7,  # Placeholder
            "evidence_nodes": await self._find_evidence_nodes(hypothesis, depth),
            "reasoning_chain": await self._generate_reasoning_chain(hypothesis, depth)
        }

    async def _analyze_domain_coverage(self) -> Dict[str, Any]:
        """Analyze domain coverage in the knowledge graph"""
        entity_types = defaultdict(int)
        for _, data in self.knowledge_graph.nodes(data=True):
            entity_type = data.get("type", "unknown")
            entity_types[entity_type] += 1

        return {
            "entity_type_distribution": dict(entity_types),
            "coverage_score": len(entity_types) / 10,  # Normalized score
            "primary_domains": sorted(entity_types.items(), key=lambda x: x[1], reverse=True)[:3]
        }

    async def _assess_knowledge_completeness(self) -> float:
        """Assess completeness of knowledge in the graph"""
        if len(self.knowledge_graph.nodes()) == 0:
            return 0.0

        # Simple completeness metric based on connectivity
        isolated_nodes = len(list(nx.isolates(self.knowledge_graph)))
        connectivity_ratio = 1 - (isolated_nodes / len(self.knowledge_graph.nodes()))

        # Consider relationship diversity
        relationship_types = set()
        for _, _, data in self.knowledge_graph.edges(data=True):
            relationship_types.add(data.get("type", "unknown"))

        diversity_ratio = len(relationship_types) / len(self.relationship_patterns) if self.relationship_patterns else 0

        return (connectivity_ratio * 0.7 + diversity_ratio * 0.3)

    async def _extract_agent_contribution(self, data: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
        """Extract key contribution from agent data"""
        return {
            "agent": agent_name,
            "contribution_type": await self._classify_contribution_type(data),
            "key_insights": await self._extract_key_insights(data),
            "data_volume": len(str(data)),
            "timestamp": datetime.now().isoformat()
        }

    async def _analyze_cross_agent_patterns(self, agent_contributions: Dict[str, List[Dict[str, Any]]]) -> List[
        Dict[str, Any]]:
        """Analyze patterns across agent contributions"""
        patterns = []

        # Find complementary contributions
        agent_types = {}
        for agent, contributions in agent_contributions.items():
            if contributions:
                agent_types[agent] = contributions[0]["contribution_type"]

        # Identify complementary patterns
        type_groups = defaultdict(list)
        for agent, contrib_type in agent_types.items():
            type_groups[contrib_type].append(agent)

        for contrib_type, agents in type_groups.items():
            if len(agents) > 1:
                patterns.append({
                    "pattern_type": "complementary_contributions",
                    "description": f"Multiple agents ({', '.join(agents)}) provide {contrib_type} contributions",
                    "agents_involved": agents,
                    "contribution_type": contrib_type
                })

        return patterns

    async def _identify_collaboration_opportunities(self, agent_contributions: Dict[str, List[Dict[str, Any]]]) -> List[
        Dict[str, Any]]:
        """Identify opportunities for agent collaboration"""
        opportunities = []

        # Simple collaboration detection based on contribution types
        contribution_types = set()
        for contributions in agent_contributions.values():
            for contrib in contributions:
                contribution_types.add(contrib["contribution_type"])

        if len(contribution_types) >= 3:
            opportunities.append({
                "type": "cross_domain_synthesis",
                "description": "Multiple contribution types detected - opportunity for synthesis",
                "involved_contribution_types": list(contribution_types)
            })

        return opportunities

    async def _identify_cluster_theme(self, cluster_entities: List[Dict[str, Any]]) -> str:
        """Identify the theme of a cluster"""
        if not cluster_entities:
            return "unknown"

        # Simple theme detection based on entity types and names
        type_counts = defaultdict(int)
        common_words = defaultdict(int)

        for entity in cluster_entities:
            entity_type = entity.get("type", "unknown")
            type_counts[entity_type] += 1

            # Extract common words from entity names
            name = entity.get("name", "").lower()
            words = name.split()
            for word in words:
                if len(word) > 3:  # Only consider substantial words
                    common_words[word] += 1

        # Determine dominant type
        dominant_type = max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else "unknown"

        # Get most common word
        common_word = max(common_words.items(), key=lambda x: x[1])[0] if common_words else ""

        return f"{dominant_type}_{common_word}" if common_word else dominant_type

    async def _explore_reasoning_paths(self, start_entity: str, max_depth: int) -> List[Dict[str, Any]]:
        """Explore reasoning paths from a starting entity"""
        paths = []

        try:
            # Perform BFS up to max_depth
            visited = set()
            queue = [(start_entity, [start_entity], 0)]

            while queue and len(paths) < 10:  # Limit number of paths
                current, path, depth = queue.pop(0)

                if depth >= max_depth:
                    continue

                for neighbor in self.knowledge_graph.neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_path = path + [neighbor]

                        # Only consider paths with at least 2 entities
                        if len(new_path) >= 2:
                            path_description = await self._describe_path(new_path)
                            paths.append(path_description)

                        if depth + 1 < max_depth:
                            queue.append((neighbor, new_path, depth + 1))

        except:
            pass

        return paths

    async def _describe_path(self, path: List[str]) -> Dict[str, Any]:
        """Describe a path through the knowledge graph"""
        path_description = []

        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]

            edge_data = self.knowledge_graph.get_edge_data(source, target)
            if edge_data:
                # Take the first edge data
                first_edge = next(iter(edge_data.values()))
                relationship = first_edge.get("type", "related_to")

                source_data = self.knowledge_graph.nodes[source]
                target_data = self.knowledge_graph.nodes[target]

                path_description.append({
                    "step": i + 1,
                    "source": source_data.get("name", source),
                    "target": target_data.get("name", target),
                    "relationship": relationship,
                    "reasoning": f"{source_data.get('name', source)} {relationship} {target_data.get('name', target)}"
                })

        return {
            "path_length": len(path),
            "entities_involved": [self.knowledge_graph.nodes[node].get("name", node) for node in path],
            "step_by_step": path_description,
            "summary": " → ".join([self.knowledge_graph.nodes[node].get("name", node) for node in path])
        }

    async def _classify_contribution_type(self, data: Dict[str, Any]) -> str:
        """Classify the type of agent contribution"""
        data_str = str(data).lower()

        if any(term in data_str for term in ["analysis", "statistics", "metrics"]):
            return "analytical"
        elif any(term in data_str for term in ["recommendation", "suggestion", "advice"]):
            return "advisory"
        elif any(term in data_str for term in ["data", "information", "facts"]):
            return "informational"
        elif any(term in data_str for term in ["prediction", "forecast", "simulation"]):
            return "predictive"
        else:
            return "general"

    async def _extract_key_insights(self, data: Dict[str, Any]) -> List[str]:
        """Extract key insights from agent data"""
        insights = []

        # Simple insight extraction
        if isinstance(data, dict):
            for key, value in data.items():
                if key in ["summary", "insight", "finding", "conclusion"] and value:
                    insights.append(str(value))

        # Limit to 3 insights
        return insights[:3]

    async def _infer_transitive_relationships(self, node: str, depth: int) -> List[Dict[str, Any]]:
        """Infer transitive relationships for a node"""
        inferences = []

        try:
            # Find two-hop relationships
            for neighbor1 in self.knowledge_graph.neighbors(node):
                for neighbor2 in self.knowledge_graph.neighbors(neighbor1):
                    if neighbor2 != node and not self.knowledge_graph.has_edge(node, neighbor2):
                        # Potential transitive relationship
                        inference = {
                            "source": node,
                            "target": neighbor2,
                            "inferred_relationship": "transitively_related",
                            "path": [node, neighbor1, neighbor2],
                            "confidence": 0.6
                        }
                        inferences.append(inference)
        except:
            pass

        return inferences

    async def _check_hypothesis_support(self, hypothesis: str) -> bool:
        """Check if hypothesis is supported by the knowledge graph"""
        # Simplified hypothesis checking
        hypothesis_lower = hypothesis.lower()

        # Check if hypothesis terms exist in graph
        hypothesis_terms = hypothesis_lower.split()
        found_terms = 0

        for term in hypothesis_terms:
            if len(term) > 3:  # Only substantial terms
                for node_id, node_data in self.knowledge_graph.nodes(data=True):
                    if (term in node_data.get("name", "").lower() or
                            term in str(node_data.get("value", "")).lower()):
                        found_terms += 1
                        break

        # Consider hypothesis supported if at least 50% of terms are found
        return found_terms >= len(hypothesis_terms) * 0.5

    async def _find_evidence_nodes(self, hypothesis: str, depth: int) -> List[str]:
        """Find nodes that provide evidence for hypothesis"""
        evidence = []
        hypothesis_terms = hypothesis.lower().split()

        for term in hypothesis_terms:
            if len(term) > 3:
                for node_id, node_data in self.knowledge_graph.nodes(data=True):
                    if (term in node_data.get("name", "").lower() or
                            term in str(node_data.get("value", "")).lower()):
                        evidence.append(node_data.get("name", node_id))
                        break

        return evidence[:5]  # Limit to 5 evidence nodes

    async def _generate_reasoning_chain(self, hypothesis: str, depth: int) -> List[str]:
        """Generate a reasoning chain for hypothesis"""
        reasoning_chain = [
            f"Hypothesis: {hypothesis}",
            "Checking knowledge graph for supporting evidence...",
            f"Found {len(self.knowledge_graph.nodes())} entities and {len(self.knowledge_graph.edges())} relationships",
            "Analyzing entity relationships and patterns...",
            "Conclusion: Hypothesis evaluation complete"
        ]

        return reasoning_chain
# # /agents/a_series/context_agent.py
# from core.agent_base import RevenantAgentBase
# import asyncio
# from typing import Dict, List, Any
# from datetime import datetime
#
#
# class ContextAgent(RevenantAgentBase):
#     def __init__(self):
#         super().__init__(
#             name="ContextAgent",
#             description="Maintains conversation context, manages session state, and provides contextual awareness for other agents."
#         )
#         self.session_contexts = {}
#         self.conversation_history = {}
#
#     async def setup(self):
#         # Initialize context storage
#         self.session_contexts = {}
#         self.conversation_history = {}
#         await asyncio.sleep(0.1)
#
#     async def run(self, input_data: dict):
#         try:
#             session_id = input_data.get("sessionId", "default")
#             user_input = input_data.get("userInput", "")
#
#             # Update conversation history
#             await self._update_conversation_history(session_id, user_input)
#
#             # Analyze context and extract entities
#             context_analysis = await self._analyze_context(session_id, user_input)
#
#             # Maintain session state
#             session_state = await self._maintain_session_state(session_id, context_analysis)
#
#             # Generate contextual response
#             contextual_data = await self._generate_contextual_data(session_id, context_analysis)
#
#             result = {
#                 "session_id": session_id,
#                 "conversation_turn": len(self.conversation_history.get(session_id, [])),
#                 "context_entities": context_analysis["entities"],
#                 "session_state": session_state,
#                 "context_summary": contextual_data["summary"],
#                 "suggested_next_actions": contextual_data["suggestions"],
#                 "context_timestamp": datetime.now().isoformat()
#             }
#
#             return {
#                 "agent": self.name,
#                 "status": "ok",
#                 "summary": f"Context maintained for session {session_id} - {len(context_analysis['entities'])} entities identified",
#                 "data": result
#             }
#
#         except Exception as e:
#             return await self.on_error(e)
#
#     async def _update_conversation_history(self, session_id: str, user_input: str):
#         if session_id not in self.conversation_history:
#             self.conversation_history[session_id] = []
#
#         self.conversation_history[session_id].append({
#             "timestamp": datetime.now(),
#             "user_input": user_input,
#             "turn_number": len(self.conversation_history[session_id]) + 1
#         })
#
#         # Keep only last 50 messages per session
#         if len(self.conversation_history[session_id]) > 50:
#             self.conversation_history[session_id] = self.conversation_history[session_id][-50:]
#
#     async def _analyze_context(self, session_id: str, current_input: str) -> Dict[str, Any]:
#         # Simple entity extraction (in production, use NLP libraries)
#         entities = []
#         input_lower = current_input.lower()
#
#         # Extract potential entities based on keywords
#         entity_categories = {
#             "technical": ["code", "api", "database", "server", "bug", "error"],
#             "creative": ["write", "create", "design", "image", "content"],
#             "research": ["search", "find", "research", "information", "data"],
#             "commercial": ["buy", "product", "price", "money", "affiliate"],
#             "social": ["post", "share", "social", "media", "tweet"]
#         }
#
#         for category, keywords in entity_categories.items():
#             if any(keyword in input_lower for keyword in keywords):
#                 entities.append({
#                     "type": category,
#                     "confidence": 0.8,
#                     "source": "keyword_match"
#                 })
#
#         # Analyze conversation flow
#         conversation = self.conversation_history.get(session_id, [])
#         recent_topic = await self._extract_recent_topic(conversation)
#
#         return {
#             "entities": entities,
#             "recent_topic": recent_topic,
#             "conversation_length": len(conversation),
#             "topic_consistency": await self._check_topic_consistency(conversation, current_input)
#         }
#
#     async def _maintain_session_state(self, session_id: str, context_analysis: dict) -> Dict[str, Any]:
#         if session_id not in self.session_contexts:
#             self.session_contexts[session_id] = {
#                 "created_at": datetime.now(),
#                 "last_activity": datetime.now(),
#                 "interaction_count": 0,
#                 "preferred_agent": None,
#                 "topic_history": []
#             }
#
#         session = self.session_contexts[session_id]
#         session["last_activity"] = datetime.now()
#         session["interaction_count"] += 1
#
#         # Update preferred agent based on context
#         if context_analysis["entities"]:
#             primary_entity = context_analysis["entities"][0]
#             agent_mapping = {
#                 "technical": "DevAgent",
#                 "creative": "WriterAgent",
#                 "research": "SearchAgent",
#                 "commercial": "MoneyAgent",
#                 "social": "PostAgent"
#             }
#             session["preferred_agent"] = agent_mapping.get(primary_entity["type"])
#
#         # Update topic history
#         if context_analysis["recent_topic"]:
#             session["topic_history"].append({
#                 "topic": context_analysis["recent_topic"],
#                 "timestamp": datetime.now()
#             })
#
#         return session
#
#     async def _generate_contextual_data(self, session_id: str, context_analysis: dict) -> Dict[str, Any]:
#         session_state = self.session_contexts.get(session_id, {})
#
#         summary = f"Session active for {session_state.get('interaction_count', 0)} interactions"
#         if session_state.get("preferred_agent"):
#             summary += f", prefers {session_state['preferred_agent']}"
#
#         suggestions = []
#         if context_analysis["conversation_length"] > 10:
#             suggestions.append("Consider offering to summarize the conversation")
#         if len(context_analysis["entities"]) > 1:
#             suggestions.append("Multiple contexts detected - suggest focusing on primary topic")
#         if not suggestions:
#             suggestions.append("Continue natural conversation flow")
#
#         return {
#             "summary": summary,
#             "suggestions": suggestions
#         }
#
#     async def _extract_recent_topic(self, conversation: List[Dict]) -> str:
#         if not conversation:
#             return "new_conversation"
#
#         # Simple topic extraction from recent messages
#         recent_messages = [msg["user_input"] for msg in conversation[-3:]]  # Last 3 messages
#         combined_text = " ".join(recent_messages).lower()
#
#         topic_keywords = {
#             "technical": ["code", "programming", "api", "bug", "error"],
#             "writing": ["write", "content", "article", "blog", "copy"],
#             "research": ["search", "find", "information", "data"],
#             "design": ["design", "image", "create", "visual"],
#             "social": ["post", "share", "social media", "tweet"]
#         }
#
#         for topic, keywords in topic_keywords.items():
#             if any(keyword in combined_text for keyword in keywords):
#                 return topic
#
#         return "general_conversation"
#
#     async def _check_topic_consistency(self, conversation: List[Dict], current_input: str) -> float:
#         if len(conversation) < 2:
#             return 1.0  # New conversation, perfectly consistent
#
#         recent_topic = await self._extract_recent_topic(conversation[-3:])
#         current_topic = await self._extract_recent_topic([{"user_input": current_input}])
#
#         return 1.0 if recent_topic == current_topic else 0.5

# /agents/a_series/context_agent.py
from core.agent_base import RevenantAgentBase
import asyncio
from typing import Dict,  Any
from datetime import datetime


class ContextAgent(RevenantAgentBase):
    """
    Maintains conversation context, manages session state, and provides contextual awareness.

    Input:
        - sessionId (str): Session identifier
        - userInput (str): User input to analyze for context

    Output:
        - session_id (str): Session identifier
        - conversation_turn (int): Current conversation turn number
        - context_entities (list): Extracted entities from context
        - session_state (dict): Current session state
        - context_summary (str): Summary of conversation context
    """

    metadata = {
        "name": "ContextAgent",
        "series": "a_series",
        "version": "0.1.0",
        "description": "Maintains conversation context, manages session state, and provides contextual awareness for other agents."
    }

    def __init__(self):
        super().__init__(
            name="ContextAgent",
            description="Maintains conversation context, manages session state, and provides contextual awareness for other agents."
        )
        self.session_contexts = {}
        self.conversation_history = {}
        self.metadata = ContextAgent.metadata

    async def setup(self):
        # Initialize context storage
        self.session_contexts = {}
        self.conversation_history = {}
        await asyncio.sleep(0.1)

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            session_id = input_data.get("sessionId", "default")
            user_input = input_data.get("userInput", "")

            # Update conversation history
            await self._update_conversation_history(session_id, user_input)

            # Analyze context and extract entities
            context_analysis = await self._analyze_context(session_id, user_input)

            # Maintain session state
            session_state = await self._maintain_session_state(session_id, context_analysis)

            # Generate contextual response
            contextual_data = await self._generate_contextual_data(session_id, context_analysis)

            result = {
                "session_id": session_id,
                "conversation_turn": len(self.conversation_history.get(session_id, [])),
                "context_entities": context_analysis["entities"],
                "session_state": session_state,
                "context_summary": contextual_data["summary"],
                "suggested_next_actions": contextual_data["suggestions"],
                "context_timestamp": datetime.now().isoformat()
            }

            return {
                "agent": self.name,
                "status": "ok",
                "summary": f"Context maintained for session {session_id} - {len(context_analysis['entities'])} entities identified",
                "data": result
            }

        except Exception as e:
            return await self.on_error(e)

# ... existing code ...
"""
Multi-Agent Travel RAG System using CrewAI
This module defines agents for retrieval, summarization, and response composition
without requiring API keys - uses local/mock implementations
"""


import os

import sys
from typing import List, Dict, Any, Optional
import json
import re
from datetime import datetime
from collections import Counter

# Import the indexer from your existing file
try:
    from index_and_retrieve import TravelRAGIndexer, create_rag_tool_function

    INDEXER_AVAILABLE = True
except ImportError:
    print("Warning: index_and_retrieve.py not found")
    INDEXER_AVAILABLE = False

# CrewAI imports with fallback
try:
    from crewai import Agent, Task, Crew
    from crewai.tools import BaseTool, tool

    CREWAI_AVAILABLE = True
except ImportError:
    print("CrewAI not available - using fallback agent system")
    CREWAI_AVAILABLE = False


# Enhanced MockLLM for local operation without API keys
class MockLLM:
    """Enhanced Mock LLM that generates more specific responses"""

    def __init__(self):
        self.name = "MockLLM"

    def generate_response(self, prompt: str, context: str = "", role: str = "assistant") -> str:
        """Generate response based on role and context"""
        if "summarize" in prompt.lower() or role == "summarizer":
            return self._generate_summary(context, prompt)
        elif "compose" in prompt.lower() or role == "composer":
            return self._generate_composition(context, prompt)
        elif "retrieve" in prompt.lower() or role == "retriever":
            return self._generate_retrieval_response(prompt)
        else:
            return self._generate_general_response(context, prompt)

    def _generate_summary(self, context: str, prompt: str) -> str:
        """Generate detailed summary from retrieved reviews"""
        if not context:
            return "No reviews found to summarize."

        # Extract detailed information
        ratings = re.findall(r'Rating: (\d)/5', context)
        positive_phrases = re.findall(r'(?:great|excellent|perfect|wonderful|nice|good|amazing|fantastic)\s+\w+', context, re.I)
        negative_phrases = re.findall(r'(?:poor|terrible|awful|horrible|bad|disappointing|worst)\s+\w+', context, re.I)
        specific_complaints = re.findall(r'(?:noisy|dirty|rude|broken|small|uncomfortable|outdated)\s*\w*', context, re.I)

        summary = "📊 **Review Analysis Summary**:\n"

        # Add rating analysis
        if ratings:
            avg_rating = sum(int(r) for r in ratings) / len(ratings)
            summary += f"• **Average Rating**: {avg_rating:.1f}/5 stars from {len(ratings)} reviews\n"

            # Add rating distribution
            rating_counts = Counter(ratings)
            summary += "• **Rating Breakdown**: "
            for r in sorted(rating_counts.keys(), reverse=True):
                summary += f"{r}★({rating_counts[r]}) "
            summary += "\n"

        # Enhanced positive/negative analysis
        if positive_phrases:
            # Clean and count positive mentions
            cleaned_positives = [phrase.strip().title() for phrase in positive_phrases]
            common_positives = Counter(cleaned_positives).most_common(5)
            summary += "• **Guest Highlights**: "
            for phrase, count in common_positives:
                if count > 1:
                    summary += f"{phrase}({count}x) "
                else:
                    summary += f"{phrase} "
            summary += "\n"

        if negative_phrases or specific_complaints:
            # Combine and clean negative mentions
            all_negatives = negative_phrases + specific_complaints
            cleaned_negatives = [phrase.strip().title() for phrase in all_negatives if phrase.strip()]
            if cleaned_negatives:
                common_negatives = Counter(cleaned_negatives).most_common(3)
                summary += "• **Common Concerns**: "
                for phrase, count in common_negatives:
                    if count > 1:
                        summary += f"{phrase}({count}x) "
                    else:
                        summary += f"{phrase} "
                summary += "\n"

        # Add query-specific insights with better detection
        query_lower = prompt.lower()

        # Enhanced noise analysis
        if 'noise' in query_lower or 'sound' in query_lower or 'quiet' in query_lower:
            noise_mentions = len(re.findall(r'nois(e|y)|loud|sound|quiet', context, re.I))
            sound_issues = len(re.findall(r'nois(e|y)|loud|disturb', context, re.I))
            if noise_mentions > 0:
                summary += f"• **Sound Quality**: {noise_mentions} total mentions ({sound_issues} about disturbances)\n"

        # Enhanced parking analysis
        if 'parking' in query_lower:
            parking_mentions = len(re.findall(r'parking|valet', context, re.I))
            parking_cost = len(re.findall(r'parking.*(expensive|cost|fee|\$)', context, re.I))
            parking_issues = len(re.findall(r'parking.*(difficult|hard|problem|limited)', context, re.I))
            if parking_mentions > 0:
                summary += f"• **Parking Info**: {parking_mentions} mentions"
                if parking_cost > 0:
                    summary += f" ({parking_cost} about costs"
                if parking_issues > 0:
                    summary += f", {parking_issues} about availability issues"
                if parking_cost > 0 or parking_issues > 0:
                    summary += ")"
                summary += "\n"

        # Enhanced cleanliness analysis
        if 'clean' in query_lower or 'hygiene' in query_lower:
            clean_mentions = len(re.findall(r'clean|neat|tidy|spotless', context, re.I))
            dirty_mentions = len(re.findall(r'dirty|unclean|messy|filthy', context, re.I))
            if clean_mentions > 0 or dirty_mentions > 0:
                summary += f"• **Cleanliness Reports**: {clean_mentions} positive vs {dirty_mentions} negative mentions\n"

        # Service quality analysis
        if 'service' in query_lower or 'staff' in query_lower:
            service_positive = len(re.findall(r'(?:great|excellent|friendly|helpful|professional)\s*(?:service|staff)', context, re.I))
            service_negative = len(re.findall(r'(?:poor|rude|unhelpful|slow)\s*(?:service|staff)', context, re.I))
            if service_positive > 0 or service_negative > 0:
                summary += f"• **Service Quality**: {service_positive} positive vs {service_negative} negative mentions\n"

        return summary.strip()

    def _generate_composition(self, context: str, prompt: str) -> str:
        """Generate final travel recommendation - ENHANCED VERSION"""
        # Extract user query from the prompt
        user_query = prompt.split("Create detailed advice for '")[1].split("' using this analysis:")[
            0] if "Create detailed advice for '" in prompt else prompt

        if "hotel" in user_query.lower() or any(
                word in user_query.lower() for word in ['stay', 'room', 'accommodation']):
            return self._compose_hotel_recommendation(context, user_query)
        else:
            return self._compose_general_travel_advice(context, user_query)

    def _compose_hotel_recommendation(self, context: str, user_query: str) -> str:
        """Compose detailed hotel recommendation - ENHANCED VERSION"""

        # Extract key metrics from SUMMARY
        ratings = re.findall(r'Average Rating\*\*: ([\d\.]+)/5 stars from (\d+) reviews', context)
        if ratings:
            avg_rating = float(ratings[0][0])
            review_count = int(ratings[0][1])
        else:
            avg_rating = 0.0
            review_count = 0

        rec = f"🏨 **Hotel Recommendation Based on Your Query**\n"
        rec += f"*Query: {user_query}*\n\n"

        if review_count > 0:
            rec += f"📊 **Overview**: {review_count} guest reviews analyzed, {avg_rating:.1f}/5 average rating\n\n"

            # Extract and display specific insights more clearly
            insights_added = False

            if 'Sound Quality:' in context:
                sound_match = re.search(r'Sound Quality\*\*: (\d+) total mentions \((\d+) about disturbances\)', context)
                if sound_match:
                    total = int(sound_match.group(1))
                    issues = int(sound_match.group(2))
                    rec += f"🔊 **Noise Assessment**: {total} sound-related mentions, {issues} reported disturbances\n"
                    insights_added = True

            if 'Parking Info:' in context:
                parking_match = re.search(r'Parking Info\*\*: (\d+) mentions.*?(?:\(([^)]+)\))?', context)
                if parking_match:
                    mentions = int(parking_match.group(1))
                    details = parking_match.group(2) if parking_match.group(2) else "general mentions"
                    rec += f"🚗 **Parking Details**: {mentions} mentions - {details}\n"
                    insights_added = True

            if 'Cleanliness Reports:' in context:
                clean_match = re.search(r'Cleanliness Reports\*\*: (\d+) positive vs (\d+) negative', context)
                if clean_match:
                    positive = int(clean_match.group(1))
                    negative = int(clean_match.group(2))
                    rec += f"🧹 **Cleanliness Score**: {positive} positive vs {negative} negative reports\n"
                    insights_added = True

            if 'Service Quality:' in context:
                service_match = re.search(r'Service Quality\*\*: (\d+) positive vs (\d+) negative', context)
                if service_match:
                    positive = int(service_match.group(1))
                    negative = int(service_match.group(2))
                    rec += f"👥 **Service Rating**: {positive} positive vs {negative} negative experiences\n"
                    insights_added = True

            if insights_added:
                rec += "\n"

            # Enhanced recommendation based on rating
            rec += "💡 **Our Assessment**:\n"
            if avg_rating >= 4.5:
                rec += "⭐ **Excellent Choice** - Outstanding guest satisfaction\n"
                rec += "• Consistently high ratings across multiple aspects\n"
                rec += "• Guests frequently exceed expectations\n"
                rec += "• Minimal complaints reported\n"
            elif avg_rating >= 4.0:
                rec += "✅ **Highly Recommended** - Strong overall performance\n"
                rec += "• Most guests report positive experiences\n"
                rec += "• Good value for money indicated\n"
                rec += "• Minor issues are typically well-handled\n"
            elif avg_rating >= 3.0:
                rec += "⚖️ **Mixed Results** - Research recommended\n"
                rec += "• Guest experiences vary significantly\n"
                rec += "• Some aspects praised, others need attention\n"
                rec += "• Check recent reviews for current status\n"
            else:
                rec += "⚠️ **Consider Alternatives** - Multiple concerns reported\n"
                rec += "• Consistent patterns of guest dissatisfaction\n"
                rec += "• Several operational issues documented\n"
                rec += "• May not meet expectations\n"

            # Add personalized tips based on query content
            rec += "\n🎯 **Personalized Tips**:\n"
            query_lower = user_query.lower()

            if 'clean' in query_lower:
                rec += "• **For cleanliness**: Request recently renovated rooms on higher floors\n"
            if 'noise' in query_lower or 'quiet' in query_lower:
                rec += "• **For quiet stay**: Ask for rooms away from elevators, ice machines, and street-facing windows\n"
            if 'parking' in query_lower:
                rec += "• **For parking**: Call ahead to confirm availability and daily rates\n"
            if 'family' in query_lower:
                rec += "• **For families**: Look into connecting rooms and complimentary breakfast options\n"
            if 'business' in query_lower:
                rec += "• **For business**: Verify WiFi reliability and workspace amenities\n"
            if 'service' in query_lower:
                rec += "• **For service**: Consider booking directly with hotel for potential upgrades\n"

        else:
            rec += "❌ **Limited Data Available**\n"
            rec += "No specific reviews found matching your criteria.\n\n"
            rec += "💡 **Suggestions**:\n"
            rec += "• Try broader search terms\n"
            rec += "• Check multiple booking platforms\n"
            rec += "• Contact hotels directly for specific questions\n"

        return rec

    def _compose_general_travel_advice(self, context: str, user_query: str) -> str:
        """Compose detailed general travel advice - ENHANCED"""
        advice = f"✈️ **Travel Insights for Your Query**\n"
        advice += f"*Topic: {user_query}*\n\n"

        if context and "Average Rating:" in context:
            advice += "🔍 **Key Findings from Guest Reviews**:\n"

            # Extract highlights and concerns more precisely
            if 'Guest Highlights:' in context:
                highlights = context.split('Guest Highlights**: ')[1].split('\n')[0] if 'Guest Highlights**: ' in context else ""
                if highlights:
                    advice += f"• **What guests love**: {highlights}\n"

            if 'Common Concerns:' in context:
                concerns = context.split('Common Concerns**: ')[1].split('\n')[0] if 'Common Concerns**: ' in context else ""
                if concerns:
                    advice += f"• **Frequent issues**: {concerns}\n"

            advice += "\n💡 **Smart Travel Tips**:\n"
            query_lower = user_query.lower()

            if 'budget' in query_lower or 'cheap' in query_lower:
                advice += "• **Budget Strategy**: Compare prices 2-3 weeks before travel\n"
                advice += "• **Savings Tip**: Consider weekday stays vs weekends\n"
            if 'family' in query_lower:
                advice += "• **Family Focus**: Read reviews specifically mentioning children\n"
                advice += "• **Safety Check**: Verify kid-friendly amenities and pool safety\n"
            if 'location' in query_lower:
                advice += "• **Location Tips**: Check walking distances to your key destinations\n"
                advice += "• **Transport**: Research public transit options from the hotel\n"

            advice += "• **General Advice**: Contact hotels directly for special requests\n"
        else:
            advice += "💡 **General Travel Guidance**:\n"
            advice += "• **Research Phase**: Read reviews from multiple sources and timeframes\n"
            advice += "• **Booking Strategy**: Compare direct hotel rates with booking platforms\n"
            advice += "• **Communication**: Call hotels for specific amenity questions\n"
            advice += "• **Backup Plan**: Have alternative options identified\n"

        return advice

    def _generate_retrieval_response(self, prompt: str) -> str:
        """Generate response for retrieval queries"""
        return f"Searching for reviews related to: {prompt}"

    def _generate_general_response(self, context: str, prompt: str) -> str:
        """Generate general response with more detail"""
        if context:
            return (
                f"🔍 Analysis for: {prompt}\n\n"
                f"Here's what I found in traveler reviews:\n\n"
                f"{context}\n\n"
                "What specific aspect would you like more information about?"
            )
        else:
            return (
                f"I can help with your query about: {prompt}\n\n"
                "Try asking more specifically about:\n"
                "- Hotel locations and neighborhoods\n"
                "- Room quality and cleanliness\n"
                "- Service and amenities\n"
                "- Noise levels and quiet rooms\n"
                "- Parking and transportation options"
            )


# Fallback Agent System (when CrewAI is not available)
class FallbackAgent:
    """Fallback agent implementation when CrewAI is not available"""

    def __init__(self, role: str, goal: str, backstory: str, tools: List = None):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools or []
        self.llm = MockLLM()

    def execute(self, task_description: str, context: str = "") -> str:
        """Execute a task with enhanced context handling"""
        return self.llm.generate_response(task_description, context, self.role)


class FallbackCrew:
    """Fallback crew implementation"""

    def __init__(self, agents: List, tasks: List):
        self.agents = agents
        self.tasks = tasks
        self.results = []

    def kickoff(self, inputs: Dict = None) -> Dict:
        """Execute all tasks with better context passing"""
        context = ""
        results = []

        for i, task in enumerate(self.tasks):
            if i < len(self.agents):
                agent = self.agents[i]
                task_description = task.get('description', '') if isinstance(task, dict) else str(task)
                result = agent.execute(task_description, context)
                results.append(result)
                context += f"\n{result}"

        return {
            'final_output': results[-1] if results else "No results generated",
            'task_outputs': results
        }


# Enhanced Travel RAG Agent System
class TravelRAGAgents:
    """Main class for managing travel RAG agents with improved responses"""

    def __init__(self, indexer: Optional[TravelRAGIndexer] = None):
        self.indexer = indexer
        self.llm = MockLLM()

        # Initialize retrieval tool if indexer is available
        if self.indexer and INDEXER_AVAILABLE:
            self.rag_tool = create_rag_tool_function(self.indexer)
        else:
            self.rag_tool = None

        # Initialize agents
        self._setup_agents()

    def _setup_agents(self):
        """Setup all agents with enhanced descriptions"""

        if CREWAI_AVAILABLE and self.rag_tool:
            # Use actual CrewAI agents with more specific roles
            self.retriever_agent = Agent(
                role="Senior Travel Research Specialist",
                goal="Find the most relevant hotel reviews matching specific traveler needs",
                backstory=(
                    "With years of experience analyzing travel reviews, you excel at identifying "
                    "the most useful information for specific traveler queries. You understand "
                    "how to balance relevance with review quality."
                ),
                tools=[self._create_crewai_tool()],
                verbose=True
            )

            self.summarizer_agent = Agent(
                role="Travel Data Analyst",
                goal="Extract key insights and patterns from hotel reviews",
                backstory=(
                    "As a data scientist specializing in hospitality, you transform raw reviews "
                    "into actionable insights by identifying common themes, sentiment patterns, "
                    "and specific praise/complaints."
                ),
                verbose=True
            )

            self.composer_agent = Agent(
                role="Personal Travel Consultant",
                goal="Create personalized hotel recommendations based on review analysis",
                backstory=(
                    "A seasoned travel advisor with a talent for matching hotels to traveler needs. "
                    "You combine data insights with practical travel wisdom to provide trustworthy "
                    "recommendations."
                ),
                verbose=True
            )
        else:
            # Use enhanced fallback agents
            self.retriever_agent = FallbackAgent(
                role="retriever",
                goal="Find and retrieve relevant hotel reviews",
                backstory="Expert at finding relevant travel information with precision"
            )

            self.summarizer_agent = FallbackAgent(
                role="summarizer",
                goal="Analyze and summarize review data",
                backstory="Skilled at extracting key insights from multiple reviews"
            )

            self.composer_agent = FallbackAgent(
                role="composer",
                goal="Create actionable travel advice",
                backstory="Experienced in crafting personalized recommendations"
            )

    def _create_crewai_tool(self):
        """Create enhanced CrewAI tool for RAG retrieval"""

        @tool("hotel_review_search")
        def hotel_review_search(query: str) -> str:
            """Search hotel reviews for specific traveler concerns with precision"""
            if self.rag_tool:
                # Adjust top_k based on query type
                if 'noise' in query.lower() or 'parking' in query.lower():
                    return self.rag_tool(query, top_k=8)  # More reviews for specific issues
                return self.rag_tool(query, top_k=5)
            else:
                return f"Mock search results for: {query}"

        return hotel_review_search

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Enhanced query processing with better context handling"""

        # Step 1: Retrieve with query-specific parameters
        if self.rag_tool:
            # Adjust retrieval based on query type
            if any(word in user_query.lower() for word in ['noise', 'parking', 'clean']):
                retrieved_info = self.rag_tool(user_query, top_k=8)
            else:
                retrieved_info = self.rag_tool(user_query, top_k=5)
        else:
            retrieved_info = f"Mock context for: {user_query}"

        # Step 2: Create focused summary prompt
        summary_prompt = f"Analyze these reviews regarding '{user_query}':\n{retrieved_info}"

        summary = self.summarizer_agent.execute(summary_prompt, retrieved_info) if hasattr(self.summarizer_agent,
                                                                                           'execute') \
            else self.llm.generate_response(summary_prompt, retrieved_info, "summarizer")

        # Step 3: Create tailored recommendation
        composition_prompt = f"Create detailed advice for '{user_query}' using this analysis:\n{summary}"

        final_recommendation = self.composer_agent.execute(composition_prompt, summary) if hasattr(self.composer_agent,
                                                                                                   'execute') \
            else self.llm.generate_response(composition_prompt, summary, "composer")

        return {
            'user_query': user_query,
            'retrieved_info': retrieved_info,
            'summary': summary,
            'recommendation': final_recommendation,
            'timestamp': datetime.now().isoformat()
        }

    def process_query_with_crew(self, user_query: str) -> Dict[str, Any]:
        """Process query using CrewAI crew (if available) with enhanced tasks"""

        if not CREWAI_AVAILABLE:
            return self.process_query(user_query)

        # Define more specific tasks
        tasks = [
            Task(
                description=(
                    f"Search for hotel reviews specifically addressing: {user_query}\n"
                    "Prioritize reviews that mention related keywords and have detailed comments"
                ),
                agent=self.retriever_agent,
                expected_output="Relevant hotel reviews with ratings and specific details"
            ),
            Task(
                description=(
                    f"Analyze the reviews for patterns related to: {user_query}\n"
                    "Identify:\n- Most common praises/complaints\n- Rating distribution\n- Specific phrases mentioned"
                ),
                agent=self.summarizer_agent,
                expected_output="Structured analysis of review themes and patterns"
            ),
            Task(
                description=(
                    f"Create comprehensive travel advice for: {user_query}\n"
                    "Include:\n- Specific strengths/weaknesses\n- Actionable tips\n- Alternative options if relevant"
                ),
                agent=self.composer_agent,
                expected_output="Detailed, personalized travel recommendation"
            )
        ]

        # Create and run crew
        crew = Crew(
            agents=[self.retriever_agent, self.summarizer_agent, self.composer_agent],
            tasks=tasks,
            verbose=True
        )

        try:
            result = crew.kickoff()
            return {
                'user_query': user_query,
                'crew_result': result,
                'recommendation': result.get('final_output', 'No recommendation generated'),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"CrewAI execution error: {e}")
            return self.process_query(user_query)


def initialize_rag_system(csv_path: str = "processed_chunks.csv") -> TravelRAGAgents:
    """Initialize the complete RAG system with error handling"""

    if not INDEXER_AVAILABLE:
        print("Warning: Indexer not available, using enhanced mock system")
        return TravelRAGAgents(None)

    try:
        # Initialize indexer
        print("Initializing enhanced indexer...")
        indexer = TravelRAGIndexer()

        # Index documents
        print("Indexing documents with enhanced processing...")
        num_indexed = indexer.index_documents(csv_path)

        if num_indexed > 0:
            print(f"Successfully indexed {num_indexed} documents")
            return TravelRAGAgents(indexer)
        else:
            print("No documents indexed, using enhanced mock system")
            return TravelRAGAgents(None)

    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        print("Using enhanced mock system instead")
        return TravelRAGAgents(None)


def test_agents():
    """Test the enhanced agent system with sample queries"""
    print("Testing Enhanced Travel RAG Agent System...")

    # Initialize system
    agents = initialize_rag_system()

    # Test queries covering different aspects
    test_queries = [
        "Find a hotel with excellent cleanliness and comfortable beds",
        "What do guests say about noise levels and soundproofing?",
        "Looking for hotels with good parking options in the city center",
        "Recommend family-friendly hotels with good service"
    ]

    for query in test_queries:
        print(f"\n{'=' * 80}")
        print(f"QUERY: {query}")
        print('=' * 80)

        result = agents.process_query(query)

        print(f"\n📋 SUMMARY ANALYSIS:")
        print(result['summary'])

        print(f"\n💡 FINAL RECOMMENDATION:")
        print(result['recommendation'])


if __name__ == "__main__":
    test_agents()
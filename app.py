"""
Travel RAG Chatbot Streamlit Application
A multi-agent travel guide chatbot with ngrok integration
No API keys required - uses local/mock implementations
"""

import os

os.environ['STREAMLIT_HIDE_TAILING_WARNING'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Nuclear option for complete warning suppression
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", module="torch")
warnings.filterwarnings("ignore", module="streamlit")

# Disable PyTorch's custom class inspection
os.environ['TORCH_DISABLE_FFX'] = '1'

import streamlit as st
import pandas as pd
import sys
import json
import threading
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

# Import your existing modules
try:
    from crew_agents import initialize_rag_system, TravelRAGAgents

    AGENTS_AVAILABLE = True
except ImportError:
    st.error("❌ crew_agents.py not found! Make sure it's in the same directory.")
    AGENTS_AVAILABLE = False

try:
    from index_and_retrieve import TravelRAGIndexer

    INDEXER_AVAILABLE = True
except ImportError:
    st.error("❌ index_and_retrieve.py not found! Make sure it's in the same directory.")
    INDEXER_AVAILABLE = False

# Ngrok imports with fallback
try:
    from pyngrok import ngrok, conf
    import requests

    NGROK_AVAILABLE = True
except ImportError:
    st.warning("⚠️ pyngrok not installed. Install with: pip install pyngrok")
    NGROK_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="🏨 AI Travel Guide",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling


st.markdown("""
<style>
    /* Force better contrast for all themes */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white !important;
        text-align: center;
        margin-bottom: 2rem;
    }

    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid #e0e0e0;
    }

    .user-message {
        background-color: #1e3a8a !important;
        border-left: 4px solid #3b82f6;
        color: white !important;
    }

    .bot-message {
        background-color: #065f46 !important;
        border-left: 4px solid #10b981;
        color: white !important;
    }

    /* Force text to be visible in both themes */
    .chat-message strong {
        color: #ffffff !important;
        font-weight: bold;
    }

    .chat-message {
        color: #ffffff !important;
    }

    /* Fix timestamp visibility */
    .chat-message small {
        color: #cbd5e1 !important;
    }

    .stats-card {
        background: #374151 !important;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #6b7280;
        margin: 0.5rem 0;
        color: white !important;
    }

    .ngrok-status {
        background: #1e40af !important;
        padding: 0.75rem;
        border-radius: 5px;
        border: 2px solid #3b82f6;
        margin: 1rem 0;
        color: white !important;
    }

    .error-box {
        background: #dc2626 !important;
        padding: 1rem;
        border-radius: 5px;
        border: 2px solid #ef4444;
        color: white !important;
        margin: 1rem 0;
    }

    /* Fix main content text */
    .stMarkdown, .stText {
        color: var(--text-color) !important;
    }

    /* Ensure sidebar text is visible */
    .css-1d391kg, .css-1lcbmhc {
        color: var(--text-color) !important;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitTravelApp:
    """Main Streamlit Travel Chatbot Application"""

    def __init__(self):
        self.initialize_session_state()
        self.ngrok_tunnel = None

    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        if 'rag_system_loaded' not in st.session_state:
            st.session_state.rag_system_loaded = False

        if 'system_stats' not in st.session_state:
            st.session_state.system_stats = {}

        if 'ngrok_url' not in st.session_state:
            st.session_state.ngrok_url = None

        if 'ngrok_status' not in st.session_state:
            st.session_state.ngrok_status = "Not Started"

        # Initialize the RAG agents in session state
        if 'rag_agents' not in st.session_state:
            st.session_state.rag_agents = None

    def load_rag_system(self):
        """Load the RAG system with progress tracking"""
        if not AGENTS_AVAILABLE:
            st.error("❌ Agent system not available. Check crew_agents.py")
            return False

        try:
            with st.spinner("🔄 Loading RAG system... This may take a few minutes..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Step 1: Initialize
                status_text.text("Initializing indexer...")
                progress_bar.progress(20)

                # Step 2: Load system
                status_text.text("Loading and indexing documents...")
                progress_bar.progress(50)

                st.session_state.rag_agents = initialize_rag_system("processed_chunks.csv")

                # Step 3: Verify
                status_text.text("Verifying system...")
                progress_bar.progress(80)

                # Get system stats
                if hasattr(st.session_state.rag_agents, 'indexer') and st.session_state.rag_agents.indexer:
                    st.session_state.system_stats = st.session_state.rag_agents.indexer.get_stats()
                else:
                    st.session_state.system_stats = {
                        "total_documents": "Mock System",
                        "embedding_method": "Enhanced Mock",
                        "storage": "In-Memory",
                        "collection_name": "travel_reviews"
                    }

                progress_bar.progress(100)
                status_text.text("✅ System loaded successfully!")

                st.session_state.rag_system_loaded = True
                time.sleep(1)  # Brief pause to show completion

                return True

        except Exception as e:
            st.error(f"❌ Error loading RAG system: {str(e)}")
            st.info("💡 Using enhanced mock system instead")

            # Initialize with mock system
            st.session_state.rag_agents = initialize_rag_system()
            st.session_state.rag_system_loaded = True
            st.session_state.system_stats = {
                "total_documents": "Mock Data",
                "embedding_method": "Enhanced Mock",
                "storage": "Memory",
                "collection_name": "mock_reviews"
            }
            return True

    def setup_ngrok(self, port: int = 8501):
        """Setup ngrok tunnel with auth handling"""
        if not NGROK_AVAILABLE:
            st.warning("⚠️ Ngrok not available. Install pyngrok to enable public URL.")
            return None

        try:
            # Check for auth token
            ngrok_auth = os.getenv('NGROK_AUTHTOKEN')
            if not ngrok_auth:
                st.warning("""
                ⚠️ Ngrok requires an auth token for public URLs.
                Get one at: https://dashboard.ngrok.com/get-started/your-authtoken
                Then set it as environment variable NGROK_AUTHTOKEN
                """)
                return None

            # Configure with auth token
            ngrok.set_auth_token(ngrok_auth)

            # Kill existing tunnels
            ngrok.kill()

            # Create new tunnel
            tunnel = ngrok.connect(port, "http")
            public_url = tunnel.public_url

            st.session_state.ngrok_url = public_url
            st.session_state.ngrok_status = "Active"
            self.ngrok_tunnel = tunnel

            return public_url

        except Exception as e:
            st.error(f"❌ Ngrok setup failed: {str(e)}")
            st.session_state.ngrok_status = f"Error: {str(e)}"
            return None

    def process_user_query(self, user_input: str) -> Dict[str, Any]:
        """Process user query through RAG system"""
        if not st.session_state.rag_system_loaded or st.session_state.rag_agents is None:
            return {
                'error': 'RAG system not loaded',
                'recommendation': 'Please initialize the system first using the sidebar controls.'
            }

        try:
            with st.spinner("🤖 AI agents are analyzing your request..."):
                result = st.session_state.rag_agents.process_query(user_input)
                return result
        except Exception as e:
            return {
                'error': str(e),
                'recommendation': f'Sorry, I encountered an error: {str(e)}'
            }

    def render_header(self):
        """Render application header"""
        st.markdown("""
        <div class="main-header">
            <h1>🏨 AI Travel Guide Chatbot</h1>
            <p>Multi-Agent RAG System for Personalized Hotel Recommendations</p>
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Render sidebar with controls and stats"""
        st.sidebar.title("🎛️ System Controls")

        # System Status Section
        st.sidebar.subheader("📊 System Status")

        if st.session_state.rag_system_loaded:
            st.sidebar.success("✅ RAG System: Active")
        else:
            st.sidebar.error("❌ RAG System: Not Loaded")

        # Load/Reload System Button
        if st.sidebar.button("🔄 Load/Reload RAG System", type="primary"):
            with st.sidebar:
                success = self.load_rag_system()
                if success:
                    st.sidebar.success("✅ System loaded successfully!")
                    st.rerun()

        # System Statistics
        if st.session_state.system_stats:
            st.sidebar.subheader("📈 System Stats")
            for key, value in st.session_state.system_stats.items():
                st.sidebar.metric(
                    label=key.replace('_', ' ').title(),
                    value=str(value)
                )

        # Ngrok Section
        st.sidebar.subheader("🌐 Public Access (Ngrok)")

        if NGROK_AVAILABLE:
            col1, col2 = st.sidebar.columns(2)

            with col1:
                if st.button("🚀 Start Ngrok"):
                    with st.spinner("Setting up ngrok tunnel..."):
                        url = self.setup_ngrok()
                        if url:
                            st.success("✅ Tunnel created!")
                            st.rerun()

            with col2:
                if st.button("🛑 Stop Ngrok"):
                    if NGROK_AVAILABLE:
                        ngrok.kill()
                        st.session_state.ngrok_url = None
                        st.session_state.ngrok_status = "Stopped"
                        st.success("✅ Tunnel stopped!")
                        st.rerun()

            # Display ngrok status
            if st.session_state.ngrok_url:
                st.sidebar.markdown(f"""
                <div class="ngrok-status">
                    <strong>🌍 Public URL:</strong><br>
                    <a href="{st.session_state.ngrok_url}" target="_blank">
                        {st.session_state.ngrok_url}
                    </a>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.sidebar.info(f"Status: {st.session_state.ngrok_status}")
        else:
            st.sidebar.warning("⚠️ Install pyngrok for public access")

        # Chat Controls
        st.sidebar.subheader("💬 Chat Controls")

        if st.sidebar.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

        # Export chat history
        if st.session_state.chat_history:
            chat_data = {
                'timestamp': datetime.now().isoformat(),
                'chat_history': st.session_state.chat_history
            }

            st.sidebar.download_button(
                label="💾 Export Chat",
                data=json.dumps(chat_data, indent=2),
                file_name=f"travel_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        # Sample Queries
        st.sidebar.subheader("💡 Try These Queries")
        sample_queries = [
            "Find clean hotels with comfortable beds",
            "Hotels with good parking options",
            "Quiet hotels for business travelers",
            "Family-friendly hotels with good service",
            "Budget hotels in city center",
            "Hotels with excellent cleanliness ratings"
        ]

        for query in sample_queries:
            if st.sidebar.button(f"💬 {query}", key=f"sample_{hash(query)}"):
                # Add to chat and process
                self.add_to_chat("user", query)
                if st.session_state.rag_system_loaded and st.session_state.rag_agents is not None:
                    result = self.process_user_query(query)
                    response = result.get('recommendation', 'No recommendation available')
                    self.add_to_chat("assistant", response)
                    st.rerun()
                else:
                    st.sidebar.error("Please load the RAG system first!")

    def add_to_chat(self, role: str, content: str):
        """Add message to chat history"""
        st.session_state.chat_history.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })

    def render_chat_interface(self):
        """Render main chat interface with fixed HTML rendering"""
        st.subheader("💬 Travel Consultation Chat")

        # Display system status
        if not st.session_state.rag_system_loaded or st.session_state.rag_agents is None:
            st.markdown("""
            <div class="error-box">
                <strong>⚠️ System Not Ready</strong><br>
                Please load the RAG system using the sidebar controls before starting your consultation.
            </div>
            """, unsafe_allow_html=True)

        # Chat input
        user_input = st.chat_input(
            "Ask me about hotels: cleanliness, noise levels, parking, service, etc.",
            disabled=not st.session_state.rag_system_loaded
        )

        # Process new input
        if user_input and st.session_state.rag_system_loaded and st.session_state.rag_agents is not None:
            # Add user message
            self.add_to_chat("user", user_input)

            # Get AI response
            with st.spinner("🤖 AI agents are working..."):
                result = self.process_user_query(user_input)
                response = result.get('recommendation',
                                      'I apologize, but I cannot provide a recommendation at this time.')

                # Add assistant response
                self.add_to_chat("assistant", response)

            st.rerun()

        # Display chat history with FIXED HTML rendering
        if st.session_state.chat_history:
            st.subheader("📝 Conversation History")

            for i, message in enumerate(st.session_state.chat_history):
                # Get timestamp for display
                timestamp = message.get('timestamp', '')[:19] if message.get('timestamp') else ''

                if message['role'] == 'user':
                    # Clean the content - escape any HTML that might be in user input
                    clean_content = str(message['content']).replace('<', '&lt;').replace('>', '&gt;')

                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>👤 You:</strong><br>
                        {clean_content}
                        <div style="text-align: right; margin-top: 5px;">
                            <small style="color: #666;">{timestamp}</small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Clean the assistant content - remove any stray HTML tags
                    clean_content = str(message['content'])

                    # Remove any stray HTML fragments that might have leaked in
                    import re
                    clean_content = re.sub(r'<small[^>]*>.*?</small>', '', clean_content)
                    clean_content = re.sub(r'</div>$', '', clean_content.strip())

                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>🤖 AI Travel Guide:</strong><br>
                        {clean_content}
                        <div style="text-align: right; margin-top: 5px;">
                            <small style="color: #666;">{timestamp}</small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info(
                "👋 Welcome! Start by asking about hotel recommendations, cleanliness, noise levels, parking, or any travel-related questions.")

    def render_demo_section(self):
        """Render demo and instructions section"""
        st.subheader("🎯 How to Use This App")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 🚀 Getting Started
            1. **Load the System**: Click "Load/Reload RAG System" in the sidebar
            2. **Wait for Processing**: The system will index 20,000+ hotel reviews
            3. **Start Chatting**: Ask questions about hotels and travel
            4. **Get Recommendations**: Receive AI-powered insights

            ### 💡 What You Can Ask
            - Hotel cleanliness and room quality
            - Noise levels and soundproofing
            - Parking availability and costs
            - Service quality and staff friendliness
            - Location and accessibility
            - Family-friendly amenities
            """)

        with col2:
            st.markdown("""
            ### 🔧 Technical Features
            - **Multi-Agent System**: 3 specialized AI agents
            - **Vector Database**: Semantic search through reviews
            - **No API Keys**: Fully local operation
            - **Real-time Analysis**: Instant review processing
            - **Export Functionality**: Save your consultations

            ### 🌐 Public Access
            - Enable ngrok for public URL sharing
            - Share your travel insights with others
            - Demo-ready for presentations
            """)

    def run(self):
        """Main application runner"""
        # Render header
        self.render_header()

        # Render sidebar
        self.render_sidebar()

        # Main content area
        tab1, tab2 = st.tabs(["💬 Chat", "📖 Instructions"])

        with tab1:
            self.render_chat_interface()

        with tab2:
            self.render_demo_section()

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            🏨 AI Travel Guide | Built with Streamlit, CrewAI & Vector Search<br>
            <small>No API keys required • Fully local operation • Multi-agent RAG system</small>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main application entry point"""
    app = StreamlitTravelApp()
    app.run()


if __name__ == "__main__":
    # Check for required files
    required_files = ["crew_agents.py", "index_and_retrieve.py", "processed_chunks.csv"]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        st.error(f"❌ Missing required files: {', '.join(missing_files)}")
        st.info("Make sure all files are in the same directory as app.py")
        st.stop()

    main()


    #streamlit run app.py

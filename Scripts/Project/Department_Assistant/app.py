"""
Streamlit Frontend for Department Assistant RAG System
"""
import streamlit as st
import os
from main import run_query
from config import VECTOR_DB_PATH

# Set page configuration
st.set_page_config(
    page_title="Chemical Engineering Dept Assistant",
    page_icon="🧪",
    layout="wide"
)

# Title and description
st.title("🧪 IIT Bombay Chemical Engineering Department Assistant")
st.markdown("Ask questions about the department, faculty, research, courses, and more!")

# Sidebar for setup
with st.sidebar:
    st.header("⚙️ Setup")

    # Check if database exists
    db_exists = os.path.exists(VECTOR_DB_PATH)

    if db_exists:
        st.success("✅ Database is ready!")
        st.info("You can start asking questions.")
    else:
        st.warning("⚠️ Database not initialized")
        st.info(
            "Building the database means scraping the whole department website, "
            "which takes 30-45+ minutes. Running that inside a Streamlit session "
            "risks the session timing out partway through, so it's built from a "
            "terminal instead, using the project's rebuild.py script:"
        )
        st.code(
            'cd Scripts\\Project\\Department_Assistant\n'
            '& "..\\..\\..\\venv\\Scripts\\python.exe" rebuild.py',
            language="powershell"
        )
        st.caption(
            "Once it finishes, just restart this app (streamlit run app.py again) "
            "- it'll pick up the new database automatically."
        )

    st.markdown("---")
    st.markdown("### 📊 Stats")
    if db_exists:
        st.metric("Database Status", "Active")
    else:
        st.metric("Database Status", "Not Initialized")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This assistant uses RAG (Retrieval Augmented Generation) to answer questions about:
    - Faculty members and their research
    - Academic programs (B.Tech, M.Tech, PhD)
    - Research areas and projects
    - Department facilities
    - Announcements and events
    """)

# Main chat interface
if db_exists:
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the department..."):
        # Snapshot the conversation so far (before adding this new question) to
        # pass to run_query() as history - this is what lets follow-up questions
        # like "what about the other one?" get resolved correctly
        chat_history = list(st.session_state.messages)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = run_query(prompt, chat_history=chat_history)
                    st.markdown(response)

                    # Add assistant message to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

else:
    # Show placeholder if database not ready
    st.info("👈 Please build the database using the instructions in the sidebar to get started.")

    # Show example questions
    st.markdown("### 📝 Example Questions You Can Ask:")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        - What are the research areas in the department?
        - Who are the faculty members working on biological systems?
        - What programs does the department offer?
        - Tell me about the B.Tech curriculum
        """)

    with col2:
        st.markdown("""
        - What are the lab facilities available?
        - When is the next seminar?
        - What are the recent research publications?
        - How can I apply for PhD admission?
        """)

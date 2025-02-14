import streamlit as st
from typing import Generator
from groq import Groq
import re

# Must be the first Streamlit command
st.set_page_config(
    page_icon="💬",
    layout="wide",
    page_title="Groq Chat",
    initial_sidebar_state="expanded"
)

# JavaScript for handling folding functionality
st.markdown("""
<script>
function toggleThinking(id) {
    const content = document.getElementById('thinking-content-' + id);
    const arrow = document.getElementById('arrow-' + id);
    if (content.style.display === 'none') {
        content.style.display = 'block';
        arrow.innerHTML = '▼';
    } else {
        content.style.display = 'none';
        arrow.innerHTML = '▶';
    }
}
</script>
""", unsafe_allow_html=True)

# Custom CSS including thinking section styling
# st.markdown("""
# <style>
#     /* Base text colors */
#     .main, .sidebar {
#         color: #111827;
#     }
    
#     /* Markdown text colors */
#     .element-container, .stMarkdown {
#         color: #111827 !important;
#     }
    
#     /* Code block colors */
#     .markdown-text-container code {
#         color: #111827 !important;
#         background-color: #f3f4f6 !important;
#     }

#     /* Thinking section styling */
#     .thinking-section {
#         background-color: #1a1a1a;
#         border-radius: 8px;
#         margin: 10px 0;
#         padding: 1px 15px;
#         border-left: 4px solid #4f46e5;
#     }

#     .thinking-header {
#         color: #e5e7eb;
#         font-family: 'JetBrains Mono', monospace;
#         font-size: 0.9rem;
#         cursor: pointer;
#         display: flex;
#         align-items: center;
#         gap: 8px;
#     }

#     .thinking-arrow {
#         color: #6366f1;
#         font-size: 0.8rem;
#         transition: transform 0.2s;
#     }

#     .thinking-content {
#         color: #d1d5db;
#         font-family: 'JetBrains Mono', monospace;
#         font-size: 0.85rem;
#         line-height: 1.5;
#         padding: 10px 0;
#         border-top: 1px solid #374151;
#         margin-top: 8px;
#     }
    
#     /* Material Design-inspired styles for main area only */
#     .main .stTextInput > div > div > input,
#     .main .stTextArea > div > div > textarea {
#         background-color: #f8fafc;
#         border: 1px solid #e2e8f0;
#         border-radius: 8px;
#         padding: 0.75rem;
#         font-size: 1rem;
#         transition: all 0.2s;
#         color: #111827;
#     }
    
#     .main .stTextInput > div > div > input:focus,
#     .main .stTextArea > div > div > textarea:focus {
#         border-color: #6366f1;
#         box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
#     }
    
#     .main .stButton > button {
#         background-color: #6366f1;
#         color: white;
#         border-radius: 8px;
#         padding: 0.5rem 1rem;
#         font-weight: 500;
#         transition: all 0.2s;
#     }
    
#     .main .stButton > button:hover {
#         background-color: #4f46e5;
#         box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
#     }
    
#     /* Chat message styling */
#     .stChatMessage {
#         background-color: #ffffff;
#         border-radius: 12px;
#         padding: 1rem;
#         margin: 0.5rem 0;
#         box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
#         color: #111827;
#     }
    
#     /* Card-like containers for main area */
#     .main .css-12oz5g7 {
#         padding: 1.5rem;
#         border-radius: 12px;
#         background-color: #ffffff;
#         box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
#         margin: 0.5rem 0;
#     }
    
#     /* Improved spacing */
#     .block-container {
#         padding: 2rem;
#         max-width: 1200px;
#     }
    
#     /* Material Design typography for main area */
#     .main h1, .main h2, .main h3 {
#         font-family: 'Roboto', sans-serif;
#         font-weight: 500;
#         color: #1f2937;
#     }
    
#     .main p, .main label {
#         font-family: 'Roboto', sans-serif;
#         color: #4b5563;
#     }
    
#     /* Custom divider */
#     .custom-divider {
#         height: 1px;
#         background: #e5e7eb;
#         margin: 1rem 0;
#     }

#     /* Dialog styling */
#     .stDialog {
#         background-color: rgba(0, 0, 0, 0.5);
#     }

#     /* Sidebar styling - keeping it simple */
#     .css-1d391kg {
#         background-color: #ffffff;
#         padding: 1rem;
#     }

#     /* Sidebar headings */
#     .sidebar h1, .sidebar h2, .sidebar h3, .sidebar h4 {
#         color: #111827;
#         font-size: 1rem;
#         font-weight: 600;
#         margin-bottom: 0.5rem;
#     }

#     /* Sidebar text */
#     .sidebar p, .sidebar label {
#         color: #374151;
#         font-size: 0.875rem;
#     }

#     /* Remove Material Design from sidebar components */
#     .sidebar .stSelectbox > div > div,
#     .sidebar .stSlider > div > div,
#     .sidebar .stTextInput > div > div > input,
#     .sidebar .stTextArea > div > div > textarea {
#         border: 1px solid #e5e7eb;
#         background-color: #ffffff;
#         border-radius: 4px;
#         color: #111827;
#     }

#     /* Simple button style for sidebar */
#     .sidebar .stButton > button {
#         width: 100%;
#         background-color: #f3f4f6;
#         color: #111827;
#         border: 1px solid #e5e7eb;
#         border-radius: 4px;
#         padding: 0.5rem;
#         font-size: 0.875rem;
#         font-weight: 500;
#     }

#     .sidebar .stButton > button:hover {
#         background-color: #e5e7eb;
#     }

#     /* Ensure text visibility in all containers */
#     div[data-testid="stText"],
#     div[data-testid="stMarkdown"] {
#         color: #111827 !important;
#     }
# </style>
# """, unsafe_allow_html=True)

def format_thinking_content(content: str) -> str:
    """Format content with thinking tags into collapsible sections."""
    thinking_pattern = r'<thinking>(.*?)</thinking>'
    thinking_count = 0
    
    def replace_thinking(match):
        nonlocal thinking_count
        thinking_count += 1
        thinking_content = match.group(1).strip()
        return f'''
        <div class="thinking-section">
            <div class="thinking-header" onclick="toggleThinking({thinking_count})">
                <span id="arrow-{thinking_count}" class="thinking-arrow">▼</span>
                <span>Thinking Process</span>
            </div>
            <div id="thinking-content-{thinking_count}" class="thinking-content">
                {thinking_content}
            </div>
        </div>
        '''
    
    return re.sub(thinking_pattern, replace_thinking, content, flags=re.DOTALL)

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"],
)

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are a helpful AI assistant."

if "show_prompt_editor" not in st.session_state:
    st.session_state.show_prompt_editor = False

# Define model details
models = {
    "deepseek-r1-distill-llama-70b": {"name": "deepseek-r1-distill-llama-70b", "tokens": 32768, "developer": "deepseek"},
    "gemma2-9b-it": {"name": "gemma2-9b-it", "tokens": 8192, "developer": "Google"},
    "llama-3.3-70b-versatile": {"name": "llama-3.3-70b-versatile", "tokens": 8192, "developer": "Meta"},
    "llama-3.3-70b-specdec": {"name": "llama-3.3-70b-specdec", "tokens": 8192, "developer": "Meta"},
    "llama-3.1-8b-instant": {"name": "llama-3.1-8b-instant", "tokens": 8192, "developer": "Meta"},
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
}

# Sidebar for settings
with st.sidebar:
    st.markdown('<p style="color: #111827; font-size: 1.25rem; font-weight: 600;">Model Settings</p>', unsafe_allow_html=True)

    model_option = st.selectbox(
        "Model",
        options=list(models.keys()),
        format_func=lambda x: f"{models[x]['name']} ({models[x]['developer']})",
        index=0
    )

    # Detect model change and clear chat history if model has changed
    if st.session_state.selected_model != model_option:
        st.session_state.messages = []
        st.session_state.selected_model = model_option

    max_tokens_range = models[model_option]["tokens"]

    max_tokens = st.slider(
        "Maximum Tokens",
        min_value=512,
        max_value=max_tokens_range,
        value=min(32768, max_tokens_range),
        step=512,
        help=f"Maximum tokens for response. Current model limit: {max_tokens_range}"
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Controls response creativity. Lower values are more focused."
    )

    if st.button("Edit System Prompt"):
        st.session_state.show_prompt_editor = True

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Display current system prompt preview
    st.markdown('<p style="color: #111827; font-size: 1rem; font-weight: 600;">Current System Prompt</p>', unsafe_allow_html=True)
    st.code(st.session_state.system_prompt, language=None)

# System Prompt Editor Dialog
if st.session_state.show_prompt_editor:
    dialog = st.container()
    with dialog:
        st.markdown('<p style="color: #111827; font-size: 1.25rem; font-weight: 600;">Edit System Prompt</p>', unsafe_allow_html=True)
        
        new_system_prompt = st.text_area(
            "System Prompt",
            value=st.session_state.system_prompt,
            height=200,
            help="Define the AI assistant's behavior and role"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Cancel"):
                st.session_state.show_prompt_editor = False
                st.rerun()
        
        with col2:
            if st.button("Save", type="primary"):
                if new_system_prompt != st.session_state.system_prompt:
                    st.session_state.system_prompt = new_system_prompt
                    st.session_state.messages = []
                st.session_state.show_prompt_editor = False
                st.rerun()

# Main chat interface
st.markdown("""
<h1 style='font-size: 2.5rem; font-weight: 600; color: #1f2937; margin-bottom: 1rem;'>
    Groq Chat
</h1>
""", unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
    with st.chat_message(message["role"], avatar=avatar):
        # Format thinking tags if present
        formatted_content = format_thinking_content(message["content"])
        st.markdown(formatted_content, unsafe_allow_html=True)


def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


if prompt := st.chat_input("Message Groq..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(prompt)

    try:
        messages = [
            {"role": "system", "content": st.session_state.system_prompt}
        ] + [
            {
                "role": m["role"],
                "content": m["content"]
            }
            for m in st.session_state.messages
        ]

        chat_completion = client.chat.completions.create(
            model=model_option,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        )

        with st.chat_message("assistant", avatar="🤖"):
            chat_responses_generator = generate_chat_responses(chat_completion)
            full_response = st.write_stream(chat_responses_generator)
    except Exception as e:
        st.error(e, icon="🚨")

    if isinstance(full_response, str):
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})
    else:
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": combined_response})

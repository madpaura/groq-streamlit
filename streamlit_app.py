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

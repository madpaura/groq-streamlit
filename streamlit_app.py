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

import json
from datetime import datetime
import os

# Function to get chat label from first message
def get_chat_size_kb(messages):
    # Convert messages to string and calculate size
    import json
    messages_str = json.dumps(messages)
    return len(messages_str.encode('utf-8')) / 1024

def get_chat_label(messages, chat_id):
    if not messages:
        return f"New Chat {chat_id}"
    first_msg = messages[0]
    if first_msg["role"] == "user":
        # Truncate long messages
        label = first_msg["content"][:30]
        return f"{label}..." if len(first_msg["content"]) > 30 else label
    return f"New Chat {chat_id}"

# Function to save chats
def save_chats():
    chats_dir = "chat_history"
    os.makedirs(chats_dir, exist_ok=True)
    
    chat_data = {
        chat_id: {
            "messages": chat["messages"],
            "name": chat["name"],
            "created_at": chat["created_at"]
        }
        for chat_id, chat in st.session_state.chats.items()
    }
    
    with open(os.path.join(chats_dir, "chats.json"), "w") as f:
        json.dump(chat_data, f)

# Function to load chats
def load_chats():
    chats_dir = "chat_history"
    chats_file = os.path.join(chats_dir, "chats.json")
    
    if os.path.exists(chats_file):
        with open(chats_file, "r") as f:
            return json.load(f)
    return {}

# Initialize session state variables
if "chats" not in st.session_state:
    st.session_state.chats = load_chats()

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are a helpful AI assistant."

if "show_prompt_editor" not in st.session_state:
    st.session_state.show_prompt_editor = False

if "show_delete_confirmation" not in st.session_state:
    st.session_state.show_delete_confirmation = False

# Only set the current chat ID if there are existing chats but none is selected
if st.session_state.chats and not st.session_state.current_chat_id:
    # Select the first available chat
    st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]

# Create a new chat only if no chats exist at all (first run of the app)
if not st.session_state.chats:
    # Generate a unique chat ID
    new_chat_id = "0"
    
    st.session_state.chats[new_chat_id] = {
        "messages": [],
        "name": f"New Chat {new_chat_id}",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.current_chat_id = new_chat_id
    save_chats()

# Define model details
models = {
    "qwen-2.5-32b": {"name": "qwen-2.5-32b", "tokens": 32768, "developer": "Qwen"},
    "qwen-2.5-coder-32b": {"name": "qwen-2.5-coder-32b", "tokens": 32768, "developer": "Qwen"},
    "qwen-qwq-32b": {"name": "qwen-qwq-32b", "tokens": 32768, "developer": "Qwen"},
    "deepseek-r1-distill-llama-70b": {"name": "deepseek-r1-distill-llama-70b", "tokens": 32768, "developer": "deepseek"},
    "gemma2-9b-it": {"name": "gemma2-9b-it", "tokens": 8192, "developer": "Google"},
    "llama-3.3-70b-versatile": {"name": "llama-3.3-70b-versatile", "tokens": 8192, "developer": "Meta"},
    "llama-3.3-70b-specdec": {"name": "llama-3.3-70b-specdec", "tokens": 8192, "developer": "Meta"},
}

# Sidebar for chat management and settings
with st.sidebar:
    # Chat Management Section
    st.markdown('<p style="font-size: 1.25rem; font-weight: 600;">Chat Management</p>', unsafe_allow_html=True)
    
    # Create three columns for the buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # New Chat button (just icon)
        if st.button("➕", use_container_width=True, help="New Chat"):
            # Generate a unique chat ID that doesn't conflict with existing IDs
            existing_ids = set(st.session_state.chats.keys())
            new_chat_id = "0"
            
            # Find the first available ID that doesn't exist
            i = 0
            while new_chat_id in existing_ids:
                i += 1
                new_chat_id = str(i)
            
            st.session_state.chats[new_chat_id] = {
                "messages": [],
                "name": f"New Chat {new_chat_id}",
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.current_chat_id = new_chat_id
            save_chats()
            st.rerun()
    
    with col2:
        # Reload chats button (just icon)
        if st.button("🔄", use_container_width=True, help="Refresh Chats"):
            st.session_state.chats = load_chats()
            if not st.session_state.chats:
                # Generate a unique chat ID that doesn't conflict with existing IDs
                existing_ids = set(st.session_state.chats.keys())
                new_chat_id = "0"
                
                # Find the first available ID that doesn't exist
                i = 0
                while new_chat_id in existing_ids:
                    i += 1
                    new_chat_id = str(i)
                
                st.session_state.chats[new_chat_id] = {
                    "messages": [],
                    "name": f"New Chat {new_chat_id}",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                save_chats()
            st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
            st.rerun()
    
    with col3:
        # Delete All Chats button (just icon)
        if st.button("🗑️", use_container_width=True, help="Delete All Chats"):
            # Set a session state flag to show the confirmation dialog
            st.session_state.show_delete_confirmation = True
            st.rerun()
    
    # Handle delete confirmation outside of columns
    if st.session_state.get("show_delete_confirmation", False):
        st.warning("Are you sure you want to delete all chats? This cannot be undone.", icon="⚠️")
        confirm_col1, confirm_col2 = st.columns(2)
        with confirm_col1:
            if st.button("Yes, Delete All", key="confirm_delete_all", type="primary", use_container_width=True):
                # Create a new chat to replace the deleted ones
                new_chat_id = "0"
                st.session_state.chats = {
                    new_chat_id: {
                        "messages": [],
                        "name": f"New Chat {new_chat_id}",
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                }
                st.session_state.current_chat_id = new_chat_id
                save_chats()
                st.session_state.show_delete_confirmation = False
                st.success("All chats have been deleted.")
                st.rerun()
        with confirm_col2:
            if st.button("Cancel", key="cancel_delete_all", type="secondary", use_container_width=True):
                st.session_state.show_delete_confirmation = False
                st.rerun()
    
    # Chat History
    st.markdown('<p style="font-size: 1rem; margin-top: 1rem;">Chat History</p>', unsafe_allow_html=True)
    
    # Sort chats by creation time (newest first)
    sorted_chats = sorted(
        st.session_state.chats.items(),
        key=lambda x: x[1]["created_at"],
        reverse=True
    )
    
    for chat_id, chat_data in sorted_chats:
        # Update chat name based on first message
        chat_label = get_chat_label(chat_data["messages"], chat_id)
        
        # Create two columns for chat button and delete button
        col1, col2 = st.sidebar.columns([6, 1])
        
        # Show chat button with timestamp in the first (wider) column
        with col1:
            # Calculate chat size
            chat_size = get_chat_size_kb(chat_data["messages"])
            
            # Format button text with main label and metadata on separate lines
            button_text = f"💬 {chat_label}\n📅 {chat_data['created_at']} • {chat_size:.1f}KB"
            
            if st.button(
                button_text,
                key=f"chat_{chat_id}",
                use_container_width=True,
                type="secondary" if chat_id != st.session_state.current_chat_id else "primary"
            ):
                st.session_state.current_chat_id = chat_id
                st.rerun()
        
        # Show delete button in the second (narrower) column
        with col2:
            if st.button(
                "🗑️",
                key=f"delete_{chat_id}",
                use_container_width=True,
                type="secondary"
            ):
                # Don't delete if it's the only chat
                if len(st.session_state.chats) > 1:
                    # If deleting current chat, switch to another chat
                    if chat_id == st.session_state.current_chat_id:
                        # Find the next available chat_id
                        remaining_chats = [cid for cid in st.session_state.chats.keys() if cid != chat_id]
                        st.session_state.current_chat_id = remaining_chats[0]
                    
                    # Delete the chat
                    del st.session_state.chats[chat_id]
                    save_chats()
                    st.rerun()
                else:
                    st.sidebar.warning("Cannot delete the only chat", icon="⚠️")
    
    st.markdown('<hr style="margin: 1.5rem 0;"/>', unsafe_allow_html=True)
    
    # Model Settings Section
    st.markdown('<p style="font-size: 1.25rem; font-weight: 600;">Model Settings</p>', unsafe_allow_html=True)

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

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # System Prompt Editor in sidebar
    st.markdown('<p style="font-size: 1rem; font-weight: 600;">System Prompt</p>', unsafe_allow_html=True)
    
    # Text area for system prompt
    new_system_prompt = st.text_area(
        "System Prompt",  
        value=st.session_state.system_prompt,
        height=100,
        key="system_prompt_editor",
        help="Define the AI assistant's behavior and role",
        label_visibility="collapsed"  
    )
    
    # Save button for system prompt
    if new_system_prompt != st.session_state.system_prompt:
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Save Changes", type="primary", use_container_width=True):
                st.session_state.system_prompt = new_system_prompt
                st.rerun()
        with col2:
            if st.button("Reset", type="secondary", use_container_width=True):
                st.session_state.system_prompt = "You are a helpful AI assistant."
                st.rerun()

# Main chat interface
st.markdown("""
<h1 style='font-size: 2.5rem; font-weight: 600; color: #1f2937; margin-bottom: 1rem;'>
    LLM Chat
</h1>
""", unsafe_allow_html=True)

# Display chat messages
current_chat = st.session_state.chats[st.session_state.current_chat_id]
for message in current_chat["messages"]:
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
    current_chat = st.session_state.chats[st.session_state.current_chat_id]
    current_chat["messages"].append({"role": "user", "content": prompt})
    # Update chat name based on first message if this is the first message
    if len(current_chat["messages"]) == 1:
        current_chat["name"] = get_chat_label(current_chat["messages"], st.session_state.current_chat_id)
    save_chats()

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
            for m in current_chat["messages"]
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

    # Save the response with completion details
    completion_details = {
        "model": model_option,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "system_prompt": st.session_state.system_prompt,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if isinstance(full_response, str):
        current_chat["messages"].append({
            "role": "assistant", 
            "content": full_response,
            "completion_details": completion_details
        })
    else:
        combined_response = "\n".join(str(item) for item in full_response)
        current_chat["messages"].append({
            "role": "assistant", 
            "content": combined_response,
            "completion_details": completion_details
        })
    
    save_chats()

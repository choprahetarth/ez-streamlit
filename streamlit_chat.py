#!/usr/bin/env python3
"""
Streamlit Chat Frontend for SSE API with Context Management
Run with: streamlit run streamlit_chat.py
"""

import streamlit as st
import json
import urllib.request
import urllib.error
from typing import Generator, List, Dict

# ---------- API CONFIGURATION ----------
API_URL = "http://98.80.0.197:8003/v1/chat/completions"
MODEL_NAME = "casperhansen/llama-3.3-70b-instruct-awq"
MAX_TOKENS = 500  # Default max tokens
TOKEN = None  # Set to "sk-..." if needed

# Context management settings
MAX_CONTEXT_MESSAGES = 10  # Keep only last N messages (user + assistant pairs)
CONTEXT_WINDOW_TOKENS = 4000  # Rough estimate of model's context window

# Default headers for SSE
HEADERS = {
    "Accept": "text/event-stream",
    "Cache-Control": "no-cache",
    "Content-Type": "application/json",
}

# System prompt (developer role)
SYSTEM_PROMPT = "You are a helpful assistant. Be concise and clear in your responses."
# ---------- /API CONFIGURATION ----------


def estimate_tokens(text: str) -> int:
    """Rough estimation of tokens (4 chars ≈ 1 token)"""
    return len(text) // 4


def trim_conversation_history(messages: List[Dict], max_messages: int, max_tokens: int) -> List[Dict]:
    """
    Trim conversation history to stay within limits.
    Keeps system message and most recent exchanges.
    """
    if not messages:
        return messages
    
    # Always keep system/developer messages at the start
    system_messages = []
    conversation_messages = []
    
    for msg in messages:
        if msg["role"] in ["system", "developer"]:
            system_messages.append(msg)
        else:
            conversation_messages.append(msg)
    
    # If we have too many conversation messages, keep only the most recent ones
    if len(conversation_messages) > max_messages:
        # Keep pairs (user + assistant), so ensure even number
        keep_count = max_messages
        if keep_count % 2 == 1:
            keep_count -= 1
        conversation_messages = conversation_messages[-keep_count:]
    
    # Estimate total tokens and trim further if needed
    all_messages = system_messages + conversation_messages
    total_tokens = sum(estimate_tokens(msg["content"]) for msg in all_messages)
    
    # If still too many tokens, aggressively trim conversation
    while total_tokens > max_tokens and len(conversation_messages) > 2:
        # Remove oldest pair (user + assistant)
        conversation_messages = conversation_messages[2:]
        all_messages = system_messages + conversation_messages
        total_tokens = sum(estimate_tokens(msg["content"]) for msg in all_messages)
    
    return all_messages


def stream_response(messages: list, max_tokens: int = MAX_TOKENS) -> Generator[str, None, None]:
    """
    Stream response from the SSE API.
    Yields text chunks as they arrive.
    """
    # Trim conversation history to prevent context overflow
    trimmed_messages = trim_conversation_history(
        messages, MAX_CONTEXT_MESSAGES, CONTEXT_WINDOW_TOKENS
    )
    
    # Prepare request body
    request_body = {
        "model": MODEL_NAME,
        "messages": trimmed_messages,
        "stream": True,
        "max_completion_tokens": max_tokens,
    }
    
    # Add Authorization if provided
    headers = dict(HEADERS)
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    
    # Encode JSON body
    data_bytes = json.dumps(request_body, separators=(",", ":")).encode("utf-8")
    
    # Add timeout to prevent hanging
    req = urllib.request.Request(
        API_URL, data=data_bytes, headers=headers, method="POST"
    )
    
    try:
        # Add timeout to prevent hanging
        with urllib.request.urlopen(req, timeout=30) as resp:
            data_lines = []
            
            # Read line-by-line
            while True:
                raw = resp.readline()
                if not raw:  # connection closed
                    break
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                
                # SSE framing per spec
                if line == "":
                    # Process accumulated data lines
                    if data_lines:
                        payload = "\n".join(data_lines)
                        data_lines.clear()
                        
                        # Check for end of stream
                        if payload == "[DONE]":
                            return
                        
                        # Try to parse JSON and extract content
                        try:
                            data = json.loads(payload)
                            # Extract content from OpenAI-style response
                            if "choices" in data and len(data["choices"]) > 0:
                                choice = data["choices"][0]
                                if "delta" in choice and "content" in choice["delta"]:
                                    content = choice["delta"]["content"]
                                    if content:
                                        yield content
                        except json.JSONDecodeError:
                            pass  # Skip non-JSON lines
                    continue
                
                # Skip comments
                if line.startswith(":"):
                    continue
                
                # Collect data lines
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                    
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        yield f"❌ Error: HTTP {e.code} {e.reason}\n{body}"
    except urllib.error.URLError as e:
        yield f"❌ Connection error: {e}"
    except Exception as e:
        yield f"❌ Unexpected error: {e}"


def main():
    # Page configuration
    st.set_page_config(
        page_title="Chat with LLM",
        page_icon="💬",
        layout="wide"
    )
    
    # Title and description
    st.title("💬 LLM Chat Interface")
    st.markdown(f"Connected to: `{API_URL}`")
    st.markdown(f"Model: `{MODEL_NAME}`")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Max tokens slider
        max_tokens = st.slider(
            "Max Completion Tokens",
            min_value=50,
            max_value=2000,
            value=MAX_TOKENS,
            step=50,
            help="Maximum number of tokens to generate"
        )
        
        # Context management settings
        st.subheader("Context Management")
        max_context = st.slider(
            "Max Context Messages",
            min_value=2,
            max_value=20,
            value=MAX_CONTEXT_MESSAGES,
            step=2,
            help="Maximum number of recent messages to keep in context"
        )
        
        context_tokens = st.slider(
            "Context Window (tokens)",
            min_value=1000,
            max_value=8000,
            value=CONTEXT_WINDOW_TOKENS,
            step=500,
            help="Rough token limit for conversation context"
        )
        
        # System prompt editor
        system_prompt = st.text_area(
            "System Prompt",
            value=SYSTEM_PROMPT,
            height=100,
            help="This prompt is sent as the 'developer' role message"
        )
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # Display conversation stats
        if "messages" in st.session_state:
            user_msgs = sum(1 for m in st.session_state.messages if m["role"] == "user")
            assistant_msgs = sum(1 for m in st.session_state.messages if m["role"] == "assistant")
            total_chars = sum(len(m["content"]) for m in st.session_state.messages)
            estimated_tokens = estimate_tokens(" ".join(m["content"] for m in st.session_state.messages))
            
            st.metric("User Messages", user_msgs)
            st.metric("Assistant Messages", assistant_msgs)
            st.metric("Estimated Tokens", estimated_tokens)
            
            # Warning if approaching limits
            if estimated_tokens > context_tokens * 0.8:
                st.warning("⚠️ Approaching token limit. Consider clearing conversation.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Prepare messages for API (including system prompt)
            api_messages = [{"role": "developer", "content": system_prompt}]
            api_messages.extend(st.session_state.messages)
            
            # Stream the response
            try:
                for chunk in stream_response(api_messages, max_tokens):
                    full_response += chunk
                    # Update the placeholder with accumulated response
                    message_placeholder.markdown(full_response + "▌")
                
                # Final update without cursor
                message_placeholder.markdown(full_response)
                
            except Exception as e:
                full_response = f"❌ Error: {str(e)}"
                message_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            if full_response and not full_response.startswith("❌"):
                st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Footer with connection status
    with st.container():
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.caption("🟢 Connected to SSE API")
        with col2:
            st.caption(f"Model: {MODEL_NAME.split('/')[-1]}")
        with col3:
            st.caption(f"Max Tokens: {max_tokens}")
        with col4:
            # Show current context size
            if "messages" in st.session_state:
                current_tokens = estimate_tokens(" ".join(m["content"] for m in st.session_state.messages))
                st.caption(f"Context: {current_tokens} tokens")


# Update global variables based on sidebar settings
def update_globals():
    global MAX_CONTEXT_MESSAGES, CONTEXT_WINDOW_TOKENS
    if 'max_context' in locals():
        MAX_CONTEXT_MESSAGES = max_context
    if 'context_tokens' in locals():
        CONTEXT_WINDOW_TOKENS = context_tokens


if __name__ == "__main__":
    main()

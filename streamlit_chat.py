#!/usr/bin/env python3
"""
Streamlit Chat Frontend for SSE API
Run with: streamlit run streamlit_chat.py
"""

import streamlit as st
import json
import urllib.request
import urllib.error
from typing import Generator

# ---------- API CONFIGURATION ----------
API_URL = "http://98.80.0.197:8003/v1/chat/completions"
MODEL_NAME = "casperhansen/llama-3.3-70b-instruct-awq"
MAX_TOKENS = 500  # Default max tokens
TOKEN = None  # Set to "sk-..." if needed

# Default headers for SSE
HEADERS = {
    "Accept": "text/event-stream",
    "Cache-Control": "no-cache",
    "Content-Type": "application/json",
}

# System prompt (developer role)
SYSTEM_PROMPT = "You are a helpful assistant. Be concise and clear in your responses."
# ---------- /API CONFIGURATION ----------


def stream_response(messages: list, max_tokens: int = MAX_TOKENS) -> Generator[str, None, None]:
    """
    Stream response from the SSE API.
    Yields text chunks as they arrive.
    """
    # Prepare request body
    request_body = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": True,
        "max_completion_tokens": max_tokens,
    }
    
    # Add Authorization if provided
    headers = dict(HEADERS)
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    
    # Encode JSON body
    data_bytes = json.dumps(request_body, separators=(",", ":")).encode("utf-8")
    
    req = urllib.request.Request(
        API_URL, data=data_bytes, headers=headers, method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as resp:
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
            st.metric("User Messages", user_msgs)
            st.metric("Assistant Messages", assistant_msgs)
    
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
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption("🟢 Connected to SSE API")
        with col2:
            st.caption(f"Model: {MODEL_NAME.split('/')[-1]}")
        with col3:
            st.caption(f"Max Tokens: {max_tokens}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Streamlit chat frontend for SSE client.
Run: streamlit run streamlit_chat.py
"""

import json
import streamlit as st
import urllib.request
import urllib.error
from typing import Dict, List, Optional

# Configuration
TARGET_URL = "http://98.80.0.197:8003/v1/chat/completions"
TOKEN = None  # Set your token here if needed

# Headers for the request
HEADERS = {
    "Accept": "text/event-stream",
    "Cache-Control": "no-cache",
    "Content-Type": "application/json",
}

def send_sse_request(messages: List[Dict], model: str = "casperhansen/llama-3.3-70b-instruct-awq", max_tokens: int = 50) -> str:
    """Send SSE request and return the complete response."""
    
    # Prepare request body
    request_body = {
        "model": model,
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
        TARGET_URL, data=data_bytes, headers=headers, method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as resp:
            ctype = (resp.getheader("content-type") or "").split(";")[0].strip()
            if ctype != "text/event-stream":
                st.warning(f"Warning: response content-type is {ctype!r}, not 'text/event-stream'")
            
            data_lines = []
            event_name = None
            event_id = None
            full_response = ""
            
            # Read line-by-line until server closes
            while True:
                raw = resp.readline()
                if not raw:  # connection closed
                    break
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                
                # SSE framing per spec
                if line == "":
                    # Dispatch one event
                    if data_lines or event_name or event_id:
                        payload = "\n".join(data_lines)
                        data_lines.clear()
                        
                        if payload == "[DONE]":
                            return full_response
                        
                        # Parse the JSON response
                        try:
                            response_obj = json.loads(payload)
                            if "choices" in response_obj and len(response_obj["choices"]) > 0:
                                delta = response_obj["choices"][0].get("delta", {})
                                if "content" in delta:
                                    full_response += delta["content"]
                        except json.JSONDecodeError:
                            pass
                        
                        event_name = None
                        event_id = None
                    continue
                
                if line.startswith(":"):
                    # Comment / ping; ignore
                    continue
                
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                elif line.startswith("event:"):
                    event_name = line[6:].lstrip()
                elif line.startswith("id:"):
                    event_id = line[3:].lstrip()
            
            # Flush any trailing event
            if data_lines or event_name or event_id:
                payload = "\n".join(data_lines)
                if payload != "[DONE]":
                    try:
                        response_obj = json.loads(payload)
                        if "choices" in response_obj and len(response_obj["choices"]) > 0:
                            delta = response_obj["choices"][0].get("delta", {})
                            if "content" in delta:
                                full_response += delta["content"]
                    except json.JSONDecodeError:
                        pass
            
            return full_response
    
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        st.error(f"HTTP {e.code} {e.reason}\n{body}")
        return ""
    except urllib.error.URLError as e:
        st.error(f"Connection error: {e}")
        return ""

def main():
    st.set_page_config(
        page_title="Chat with Llama 3.3 70B",
        page_icon="��",
        layout="wide"
    )
    
    st.title("�� Chat with Llama 3.3 70B")
    st.caption("Powered by casperhansen/llama-3.3-70b-instruct-awq")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        model = st.selectbox(
            "Model",
            ["casperhansen/llama-3.3-70b-instruct-awq"],
            help="Select the model to use"
        )
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=10,
            max_value=1000,
            value=50,
            help="Maximum number of tokens to generate"
        )
        
        # Add system message option
        system_message = st.text_area(
            "System Message",
            value="Don't say vulgarities",
            help="Instructions for the AI assistant"
        )
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to chat about?"):
        # Prepare messages with system message
        messages = [{"role": "developer", "content": system_message}]
        
        # Add conversation history
        for msg in st.session_state.messages:
            messages.append(msg)
        
        # Add current user message
        messages.append({"role": "user", "content": prompt})
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = send_sse_request(messages, model, max_tokens)
            
            if response:
                st.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.error("Failed to get response from the server")
    
    # Display connection info
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Endpoint:** `{TARGET_URL}`")
    st.sidebar.markdown(f"**Messages:** {len(st.session_state.messages)}")
    
    # Add some example prompts
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Example Prompts:**")
    example_prompts = [
        "Explain quantum computing in simple terms",
        "Write a short poem about AI",
        "Help me debug this Python code",
        "What's the weather like today?"
    ]
    
    for example in example_prompts:
        if st.sidebar.button(example, key=f"example_{example}"):
            st.session_state.messages.append({"role": "user", "content": example})
            st.rerun()

if __name__ == "__main__":
    main()

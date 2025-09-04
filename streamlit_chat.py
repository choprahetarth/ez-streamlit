#!/usr/bin/env python3
"""
Stateless Streamlit Chat Frontend - Single message at a time, no history
Run with: streamlit run streamlit_chat.py
"""

import streamlit as st
import json
import httpx
from typing import Generator

# ---------- API CONFIGURATION ----------
API_URL = "http://98.80.0.197:8003/v1/chat/completions"
MODEL_NAME = "casperhansen/llama-3.3-70b-instruct-awq"
MAX_TOKENS = 40  # Default max tokens
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


def stream_response(user_message: str, system_prompt: str, max_tokens: int = MAX_TOKENS) -> Generator[str, None, None]:
    """
    Stream response from the SSE API - completely stateless, single request.
    Each call is isolated like a curl command.
    """
    # Prepare request body - just system prompt and current user message
    request_body = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "stream": True,
        "max_completion_tokens": max_tokens,
    }
    
    # Add Authorization if provided
    headers = dict(HEADERS)
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    
    # Create a completely fresh client for this single request
    transport = httpx.HTTPTransport(retries=0)
    
    with httpx.Client(
        transport=transport,
        timeout=httpx.Timeout(30.0, connect=5.0),
        limits=httpx.Limits(max_connections=1, max_keepalive_connections=0)
    ) as client:
        
        try:
            # Stream the response
            with client.stream(
                'POST', 
                API_URL, 
                json=request_body, 
                headers=headers
            ) as response:
                response.raise_for_status()
                
                data_lines = []
                
                # Process SSE stream line by line
                for line in response.iter_lines():
                    # SSE framing per spec
                    if line == "":
                        # Process accumulated data lines
                        if data_lines:
                            payload = "\n".join(data_lines)
                            data_lines.clear()
                            
                            # Check for end of stream
                            if payload == "[DONE]":
                                response.close()
                                return
                            
                            # Try to parse JSON and extract content
                            try:
                                data = json.loads(payload)
                                if "choices" in data and len(data["choices"]) > 0:
                                    choice = data["choices"][0]
                                    if "delta" in choice and "content" in choice["delta"]:
                                        content = choice["delta"]["content"]
                                        if content and content != "first chunk":  # Skip the hardcoded "first chunk"
                                            yield content
                            except json.JSONDecodeError:
                                pass
                        continue
                    
                    # Skip comments
                    if line.startswith(":"):
                        continue
                    
                    # Collect data lines
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                
                response.close()
                        
        except httpx.HTTPStatusError as e:
            yield f"❌ Error: HTTP {e.response.status_code}\n{e.response.text}"
        except httpx.ConnectTimeout:
            yield f"❌ Connection timeout: Could not connect to {API_URL}"
        except httpx.ReadTimeout:
            yield f"❌ Read timeout: Server took too long to respond"
        except httpx.RequestError as e:
            yield f"❌ Connection error: {e}"
        except Exception as e:
            yield f"❌ Unexpected error: {e}"
        finally:
            try:
                client.close()
            except:
                pass


def main():
    # Page configuration - NO SESSION STATE
    st.set_page_config(
        page_title="Stateless LLM Chat",
        page_icon="🔄",
        layout="wide"
    )
    
    # Title
    st.title("🔄 Stateless LLM Interface (Like curl)")
    st.markdown(f"**Endpoint:** `{API_URL}`")
    st.markdown(f"**Model:** `{MODEL_NAME}`")
    st.markdown("---")
    st.info("💡 Each message is sent as an isolated request. No conversation history is maintained.")
    
    # Two column layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("⚙️ Settings")
        
        # Max tokens slider
        max_tokens = st.slider(
            "Max Tokens",
            min_value=50,
            max_value=2000,
            value=MAX_TOKENS,
            step=50
        )
        
        # System prompt editor
        system_prompt = st.text_area(
            "System Prompt",
            value=SYSTEM_PROMPT,
            height=150,
            help="This is sent as 'developer' role"
        )
        
        st.divider()
        st.caption("Each request is independent")
        st.caption("No history maintained")
        st.caption("Like running curl each time")
    
    with col2:
        st.header("📝 Send Message")
        
        # Simple form for single message
        with st.form("message_form", clear_on_submit=True):
            user_input = st.text_area(
                "Your Message:",
                height=100,
                placeholder="Type your message here...",
                key="user_input"
            )
            
            col_a, col_b = st.columns([1, 4])
            with col_a:
                submit = st.form_submit_button("🚀 Send", use_container_width=True, type="primary")
            with col_b:
                st.caption("Sends a single, isolated request")
        
        # Response area
        if submit and user_input:
            st.divider()
            st.subheader("Response:")
            
            # Show the request being sent (like curl -v)
            with st.expander("📤 Request Details", expanded=False):
                st.json({
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "developer", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    "stream": True,
                    "max_completion_tokens": max_tokens
                })
            
            # Stream the response
            response_container = st.empty()
            full_response = ""
            
            try:
                for chunk in stream_response(user_input, system_prompt, max_tokens):
                    full_response += chunk
                    response_container.markdown(full_response + "▌")
                
                # Final response without cursor
                response_container.markdown(full_response)
                
                # Show completion status
                st.success("✅ Request completed")
                
            except Exception as e:
                response_container.error(f"❌ Error: {str(e)}")
                st.error("Request failed")
    
    # Footer
    st.markdown("---")
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Clear Output"):
                st.rerun()
        with col2:
            st.caption(f"Endpoint: {API_URL.split('//')[1].split('/')[0]}")
        with col3:
            st.caption("Stateless Mode")


if __name__ == "__main__":
    main()

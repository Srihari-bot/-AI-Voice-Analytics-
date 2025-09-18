def get_resolution_system_message(intent_category: str = 'general') -> str:
    """Get system message for resolution generation with RAG context"""
    return """You are a Tally support assistant. Using the provided knowledge base context, generate a structured resolution in this EXACT format:

File: [filename]
Identified Intent: [Brief description on same line]
Recommended Resolution Path:
1. [First step]
2. [Second step] 
3. [Additional steps as needed]
Suggested Knowledge Base Article:
[URL from candidate links]

Use ONLY the information provided in the context. Do not repeat sections or add extra content."""

def get_resolution_user_message_with_rag(intent: str, content: str, rag_context: dict) -> str:
    """Generate resolution prompt with RAG context"""
    
    file_name = "audio_file"  # Default, will be overridden
    context = rag_context.get("context", "")
    raw_steps = rag_context.get("raw_steps", [])
    candidate_links = rag_context.get("candidate_links", [])
    chosen_link = rag_context.get("chosen_link")
    
    steps_md = "\n".join([f"- {s}" for s in raw_steps]) if raw_steps else "None"
    final_links = [chosen_link] if chosen_link else candidate_links
    links_md = "\n".join([f"- {u}" for u in final_links]) if final_links else "None"
    
    return f"""Based on the customer issue and knowledge base context, provide a structured resolution:

Customer Issue: {intent}
Transcript: {content}

Knowledge Base Context:
{context}

Available Steps:
{steps_md}

Available Links:
{links_md}

Generate a response in the exact format specified in the system message."""

def clean_repeated_output(text: str) -> str:
    """Clean duplicate sections from LLM output"""
    if "END OF ANSWER" in text:
        text = text.split("END OF ANSWER")[0].strip()
    
    file_match = re.search(
        r'(File(?:\s*:)?\s*.+?)(?=\s*(?:END OF ANSWER|File(?:\s*:)?|$))', 
        text, re.DOTALL | re.IGNORECASE
    )
    
    if file_match:
        first_response = file_match.group(1).strip()
        if all(keyword in first_response.lower() for keyword in 
               ['file', 'identified intent', 'recommended resolution', 'suggested knowledge']):
            return first_response
    
    # Fallback logic for duplicate detection
    lines = text.split('\n')
    seen_sections = {}
    result_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        for section in ['File:', 'Identified Intent:', 'Recommended Resolution Path:', 'Suggested Knowledge Base Article:']:
            if line_stripped.startswith(section):
                if section in seen_sections:
                    return '\n'.join(result_lines).strip()
                seen_sections[section] = True
                break
        result_lines.append(line)
    
    return '\n'.join(result_lines).strip()

def format_tally_resolution(raw_output: str, filename: str) -> str:
    """Format the final resolution output with clickable links"""
    # Clean duplicates first
    cleaned = clean_repeated_output(raw_output)
    
    # Add 'File: filename' at the very beginning
    if not cleaned.startswith("File:"):
        cleaned = f"File: {filename}\n\n" + cleaned
    
    # Convert raw URLs to markdown links (THIS IS THE KEY FIX)
    url_pattern = r"(?<!\()(?<!\[)(https?://[^\s\)]+)"
    def make_clickable(match):
        url = match.group(0)
        # Remove any trailing punctuation
        url = re.sub(r'[,;.]+$', '', url)
        return f"[{url}]({url})"
    
    cleaned = re.sub(url_pattern, make_clickable, cleaned)
    
    # Format headers with markdown
    cleaned = re.sub(r'(?i)^File\s*:\s*.+$', f'**File:** {filename}', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'(?i)^Identified\s+Intent\s*:\s*', '**Identified Intent:** ', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'(?i)^Recommended\s+Resolution\s+Path\s*:\s*', '**Recommended Resolution Path:**\n\n', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'(?i)^Suggested\s+Knowledge\s+Base\s+Article\s*:\s*', '**Suggested Knowledge Base Article:**\n\n', cleaned, flags=re.MULTILINE)
    
    # Clean extra whitespace
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    # Remove citation markers
    cleaned = re.sub(r"(?:\s*\[(?:web|chart|memory|attached_file|image):\d+\])+", "", cleaned, flags=re.IGNORECASE)
    
    return cleaned.strip()



# Import required for cleaning function
import re

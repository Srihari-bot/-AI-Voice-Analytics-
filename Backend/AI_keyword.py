"""
AI Keyword & Entity Extraction Module for Tally Support System

This module provides prompts and functions for extracting keywords and entities
from transcribed audio content related to Tally ERP, GST, and business operations.
"""

def get_keyword_extraction_system_message():
    """
    Get the system message for AI keyword and entity extraction.
    """
    return """You are an AI specialist in keyword and entity extraction for Tally ERP and GST-related content. 

Your task is to analyze transcribed text and extract:
1. **Keywords**: Important technical terms, concepts, and topics
2. **Confidence Scores**: Rate each extraction from 0-100% based on relevance and clarity

EXTRACTION CATEGORIES:
- **GST Terms**: GSTR-1, GSTR-2A, GSTR-3B, GST returns, tax rates, HSN codes, etc.
- **Tally Features**: Vouchers, ledgers, reports, masters, inventory, etc.
- **Business Entities**: Company names, supplier names, invoice numbers, amounts, etc.
- **Actions**: Create, generate, file, reconcile, amend, etc.
- **Issues/Problems**: Mismatch, error, discrepancy, missing, incorrect, etc.

OUTPUT FORMAT:
Return ONLY a valid JSON object with this exact structure:
{
    "keywords": [
        {"term": "keyword", "category": "category_name", "confidence": 95},
        {"term": "another_keyword", "category": "category_name", "confidence": 88}
    ],
    "overall_confidence": 90
}

GUIDELINES:
- Extract 5-10 most relevant keywords
- Confidence scores should reflect how certain you are about the relevance
- Categories: "gst", "tally_feature", "business_entity", "action", "issue", "financial"
- Overall confidence is the average relevance to Tally/GST domain
- Only include terms that are actually relevant to business/accounting/Tally context
- Focus on understanding the context and meaning, not just pattern matching"""

def get_keyword_extraction_user_message(transcription_text):
    """
    Generate the user message for keyword and entity extraction.
    
    Args:
        transcription_text (str): The transcribed audio content
    
    Returns:
        str: Formatted user message for keyword extraction
    """
    return f"""Please analyze the following transcribed text and extract relevant keywords and entities for a Tally ERP support system:

TRANSCRIPTION:
"{transcription_text}"

Extract keywords that would help categorize and understand this support request. Focus on:
- Technical terms related to Tally, GST, or accounting
- Specific forms, reports, or features mentioned
- Business entities or data mentioned
- Actions the user wants to perform
- Problems or issues described

Return the results in the specified JSON format with confidence scores."""

def get_business_context_keywords():
    """
    Get a list of common business context keywords for reference.
    """
    return {
        "gst_terms": [
            "GST", "GSTR-1", "GSTR-2A", "GSTR-3B", "GSTR-4", "GSTR-9", 
            "input tax credit", "ITC", "output tax", "HSN code", "SAC code",
            "tax rate", "CGST", "SGST", "IGST", "CESS", "e-way bill", "e-invoice"
        ],
        "tally_features": [
            "voucher", "ledger", "stock item", "godown", "cost center", "budget",
            "bill of materials", "payroll", "TDS", "TCS", "bank reconciliation",
            "inventory", "accounts", "masters", "reports", "backup", "restore"
        ],
        "business_entities": [
            "supplier", "customer", "vendor", "party", "company", "branch",
            "invoice", "bill", "receipt", "payment", "purchase order", "sales order"
        ],
        "actions": [
            "create", "generate", "file", "submit", "reconcile", "match", "verify",
            "amend", "cancel", "delete", "modify", "update", "configure", "setup"
        ],
        "issues": [
            "mismatch", "error", "discrepancy", "missing", "incorrect", "failed",
            "timeout", "rejected", "pending", "duplicate", "invalid", "blocked"
        ]
    }

def validate_extraction_response(response_text):
    """
    Validate and clean the keyword extraction response.
    
    Args:
        response_text (str): Raw response from AI
    
    Returns:
        dict: Validated and cleaned extraction results
    """
    import json
    import re
    
    try:
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            result = json.loads(json_str)
        else:
            result = json.loads(response_text)
        
        # Validate structure
        if not isinstance(result, dict):
            raise ValueError("Response is not a dictionary")
        
        if 'keywords' not in result:
            raise ValueError("Missing required fields")
        
        # Ensure confidence scores are valid
        for keyword in result.get('keywords', []):
            if 'confidence' in keyword:
                keyword['confidence'] = max(0, min(100, int(keyword['confidence'])))
        
        # Set overall confidence if missing
        if 'overall_confidence' not in result:
            all_confidences = [k.get('confidence', 50) for k in result.get('keywords', [])]
            result['overall_confidence'] = int(sum(all_confidences) / len(all_confidences)) if all_confidences else 50
        
        return result
        
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        # Return fallback response if parsing fails
        return {
            "keywords": [
                {"term": "tally_support", "category": "tally_feature", "confidence": 70},
                {"term": "analysis_needed", "category": "action", "confidence": 60}
            ],
            "overall_confidence": 65,
            "error": f"Parsing error: {str(e)}"
        }

def extract_keywords_simple_fallback(transcription_text):
    """
    Simple keyword extraction fallback when AI service is unavailable.
    Focuses on understanding context rather than pattern matching.
    
    Args:
        transcription_text (str): The transcribed text
    
    Returns:
        dict: Simple keyword extraction results
    """
    text_lower = transcription_text.lower()
    context_keywords = get_business_context_keywords()
    
    found_keywords = []
    
    # Simple context-based matching for common terms
    for category, terms in context_keywords.items():
        for term in terms:
            if term.lower() in text_lower:
                found_keywords.append({
                    "term": term,
                    "category": category,
                    "confidence": 80
                })
    
    # Remove duplicates and limit results
    unique_keywords = []
    seen_terms = set()
    for kw in found_keywords:
        if kw['term'].lower() not in seen_terms:
            seen_terms.add(kw['term'].lower())
            unique_keywords.append(kw)
    
    # Limit to top results
    unique_keywords = unique_keywords[:10]
    
    # Calculate overall confidence
    all_confidences = [kw['confidence'] for kw in unique_keywords]
    overall_confidence = int(sum(all_confidences) / len(all_confidences)) if all_confidences else 60
    
    return {
        "keywords": unique_keywords,
        "overall_confidence": overall_confidence,
        "method": "fallback"
    }

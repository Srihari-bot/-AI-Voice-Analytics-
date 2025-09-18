import os
import re
import logging
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv

# Try to import sentence_transformers, but make it optional
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"SentenceTransformers not available: {e}")
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# DISABLE ALL TELEMETRY BEFORE CHROMADB IMPORT
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["POSTHOG_DISABLED"] = "true"

import chromadb
from chromadb.config import Settings

load_dotenv()

# Configuration  
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip()
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_kb")  # Use separate directory for knowledge base
LINKS_COLLECTION = os.getenv("LINKS_COLLECTION", "kb_links_index")
CONTENT_COLLECTION = os.getenv("CONTENT_COLLECTION", "kb_content_index")
TOP_K_LINKS = int(os.getenv("TOP_K_LINKS", "4"))
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "8"))
CONTROL_CENTRE_KEYS = ["control centre", "control-center", "control center"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TallyKnowledgeRetriever:
    def __init__(self):
        self.sbert = None
        self.chroma_client = None
        self.links_collection = None
        self.content_collection = None
        self._initialized = False
        # Don't initialize immediately to avoid conflicts
    
    def _initialize_components(self):
        try:
            # Try to initialize embedding model
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    self.sbert = SentenceTransformer(EMBED_MODEL)
                    logger.info(f"Embedding model loaded: {EMBED_MODEL}")
                except Exception as embed_error:
                    logger.warning(f"Failed to load embedding model: {str(embed_error)}")
                    logger.info("RAG will use fallback context only")
                    self.sbert = None
            else:
                logger.info("SentenceTransformers not available, using fallback context only")
                self.sbert = None
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=PERSIST_DIR,
                settings=Settings(anonymized_telemetry=False, allow_reset=True)
            )
            self.links_collection = self.chroma_client.get_or_create_collection(LINKS_COLLECTION)
            self.content_collection = self.chroma_client.get_or_create_collection(CONTENT_COLLECTION)
            logger.info("RAG components initialized without telemetry")
        except Exception as e:
            logger.error(f"Error initializing RAG components: {str(e)}")
            # Don't raise, allow fallback to work
            self.sbert = None
            self.chroma_client = None
            self.links_collection = None
            self.content_collection = None
    
    def retrieve_context(self, transcript: str) -> Dict[str, Any]:
        try:
            # Initialize components on first use
            if not self._initialized:
                self._initialize_components()
                self._initialized = True
            
            # If embedding model is not available, use fallback
            if not self.sbert or not self.chroma_client:
                logger.info("Embedding model or ChromaDB not available, using enhanced fallback context")
                return self._get_fallback_context(transcript)
            
            # Query links
            hop1_urls = self._query_links(transcript, TOP_K_LINKS)
            
            # Query content
            results = self._query_content(transcript, hop1_urls, TOP_K_CHUNKS)
            
            # Build context
            context, raw_steps, candidate_links = self._build_context_and_links(results)
            
            # Choose best link
            chosen_link = next(
                (u for u in candidate_links if any(k in u.lower() for k in CONTROL_CENTRE_KEYS)), 
                None
            ) or (candidate_links[0] if candidate_links else None)
            
            # If no knowledge base data is available, provide fallback
            if not chosen_link and not candidate_links:
                logger.info("No knowledge base data available, providing enhanced fallback content")
                return self._get_fallback_context(transcript)
            
            return {
                "context": context,
                "raw_steps": raw_steps,
                "candidate_links": candidate_links,
                "chosen_link": chosen_link,
                "hop1_urls": hop1_urls
            }
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return self._get_fallback_context(transcript)
    
    def _query_links(self, query: str, k: int) -> List[str]:
        if not self.sbert or not self.links_collection:
            return []
        
        qvec = self.sbert.encode([query]).tolist()
        res = self.links_collection.query(
            query_embeddings=qvec, 
            n_results=max(k, 20), 
            include=["metadatas", "documents"]
        )
        mets = res.get("metadatas", [[]])[0]
        urls = [md.get("source_url", "") for md in mets if isinstance(md, dict) and md.get("source_url")]
        return self._boost_urls(urls)[:k]
    
    def _query_content(self, query: str, candidate_urls: List[str], k: int) -> Dict:
        if not self.sbert or not self.content_collection:
            return {"metadatas": [[]], "documents": [[]]}
        
        qvec = self.sbert.encode([query]).tolist()
        where_filter = None
        if candidate_urls:
            valid_urls = [u.strip() for u in set(candidate_urls) if isinstance(u, str) and u and u.strip()]
            if valid_urls:
                where_filter = {"source_url": {"$in": valid_urls}}
        
        query_kwargs = {
            "query_embeddings": qvec, 
            "n_results": max(k, 50), 
            "include": ["metadatas", "documents"]
        }
        if where_filter:
            query_kwargs["where"] = where_filter
        
        res = self.content_collection.query(**query_kwargs)
        mets = res.get("metadatas", [[]])[0]
        docs = res.get("documents", [[]])[0]
        pairs = list(zip(mets, docs))
        
        pairs.sort(key=lambda p: (
            0 if any(k in (p[0].get("source_url") or "").lower() for k in CONTROL_CENTRE_KEYS) else 1,
            0 if "#" in (p[0].get("source_url") or "") else 1
        ))
        pairs = pairs[:k]
        
        return {
            "metadatas": [[p[0] for p in pairs]],
            "documents": [[p[1] for p in pairs]],
        }
    
    def _build_context_and_links(self, results: Dict) -> Tuple[str, List[str], List[str]]:
        mets = results.get("metadatas", [[]])[0]
        docs = results.get("documents", [[]])[0]
        parts, candidate_links, raw_steps = [], [], []
        
        for md, dc in zip(mets, docs):
            md = md if isinstance(md, dict) else {}
            url = md.get("source_url") or ""
            question = md.get("question") or ""
            rec_path = md.get("recommended_path") or "N/A"
            content_text = str(dc) if dc else ""
            
            if url:
                candidate_links.append(url)
                header = f"Source URL: [{url}]({url})"
            else:
                header = "Source URL: N/A"
            
            if rec_path and rec_path != "N/A":
                path_steps = [s.strip() for s in rec_path.split(">") if s.strip()]
                raw_steps.extend(path_steps)
            
            for match in re.findall(r"(?mi)^(?:-|\*|\d+\.)\s+([^\n]+)$", content_text):
                clean_step = re.sub(r"\s+", " ", match).strip(" .,:;)")
                if clean_step and len(clean_step) > 5:
                    raw_steps.append(clean_step)
            
            context_part = "\n".join([
                header,
                f"Question: {question}" if question else "",
                f"Recommended Path: {rec_path}" if rec_path != "N/A" else "",
                f"Content: {content_text[:500]}..." if len(content_text) > 500 else f"Content: {content_text}"
            ])
            parts.append(context_part)
        
        # Deduplicate
        def dedupe_preserve_order(seq):
            seen = set()
            result = []
            for item in seq:
                key = item.lower().strip()
                if key not in seen and len(key) > 2:
                    seen.add(key)
                    result.append(item)
            return result
        
        context = "\n\n---\n\n".join(parts) if parts else "No relevant information found"
        deduped_steps = dedupe_preserve_order(raw_steps)[:10]
        deduped_links = dedupe_preserve_order(candidate_links)
        
        return context, deduped_steps, deduped_links
    
    def _boost_urls(self, urls: List[str]) -> List[str]:
        def priority_key(url):
            url_lower = (url or "").lower()
            return (
                0 if any(k in url_lower for k in CONTROL_CENTRE_KEYS) else 1,
                0 if "#" in url else 1,
                len(url)
            )
        
        unique_urls = []
        seen = set()
        for url in urls:
            if url and url not in seen:
                seen.add(url)
                unique_urls.append(url)
        
        return sorted(unique_urls, key=priority_key)
    
    def _get_fallback_context(self, transcript: str) -> Dict[str, Any]:
        """Provide fallback context when no knowledge base data is available."""
        # Enhanced fallback with more specific Tally guidance
        transcript_lower = transcript.lower()
        
        # Determine context based on transcript content
        if any(keyword in transcript_lower for keyword in ['gst', 'tax', 'eway', 'bill', 'gstr']):
            context = """
            **GST Related Query Detected**
            
            For GST-related issues in Tally Prime:
            1. Go to Gateway of Tally
            2. Press Alt+G (GST Reports)
            3. Select the required GST report (GSTR-1, GSTR-3B, etc.)
            4. Configure the tax period
            5. Generate and export the report
            
            Common GST Reports:
            - GSTR-1: Outward supplies
            - GSTR-3B: Monthly return
            - GSTR-2A: Inward supplies
            - E-way Bill: For transportation
            """
            chosen_link = "https://help.tallysolutions.com/tally-prime/gst-reports/india-gst-faq-tally/#1362605"
            steps = [
                "Go to Gateway of Tally",
                "Press Alt+G (GST Reports)",
                "Select the required GST report",
                "Configure tax period",
                "Generate and export report"
            ]
            
        elif any(keyword in transcript_lower for keyword in ['ledger', 'account', 'create', 'master']):
            context = """
            **Ledger/Account Related Query Detected**
            
            For ledger creation and management in Tally Prime:
            1. Go to Gateway of Tally
            2. Press Alt+M (Masters)
            3. Select Ledger
            4. Press Alt+C (Create)
            5. Enter ledger name and details
            6. Save the ledger
            
            Types of Ledgers:
            - Sundry Debtors: For customers
            - Sundry Creditors: For suppliers
            - Cash: For cash transactions
            - Bank: For bank accounts
            """
            chosen_link = "https://help.tallysolutions.com/tally-prime/accounting/ledger-creation/#create-ledger"
            steps = [
                "Go to Gateway of Tally",
                "Press Alt+M (Masters)",
                "Select Ledger",
                "Press Alt+C (Create)",
                "Enter ledger details",
                "Save the ledger"
            ]
            
        elif any(keyword in transcript_lower for keyword in ['stock', 'inventory', 'item', 'product']):
            context = """
            **Inventory/Stock Related Query Detected**
            
            For inventory management in Tally Prime:
            1. Go to Gateway of Tally
            2. Press Alt+I (Inventory Reports)
            3. Select Stock Summary or Item Summary
            4. Configure date range if needed
            5. View the report
            
            Common Inventory Reports:
            - Stock Summary: Overall stock position
            - Item Summary: Individual item details
            - Stock Valuation: Value of inventory
            - Movement Analysis: Stock movement trends
            """
            chosen_link = "https://help.tallysolutions.com/tally-prime/inventory/inventory-reports/#stock-summary"
            steps = [
                "Go to Gateway of Tally",
                "Press Alt+I (Inventory Reports)",
                "Select Stock Summary",
                "Configure date range",
                "View the report"
            ]
            
        elif any(keyword in transcript_lower for keyword in ['voucher', 'entry', 'transaction', 'sales', 'purchase']):
            context = """
            **Voucher Entry Related Query Detected**
            
            For voucher entry in Tally Prime:
            1. Go to Gateway of Tally
            2. Press the appropriate voucher key (F8 for Sales, F9 for Purchase)
            3. Enter party name
            4. Enter item details
            5. Enter quantities and rates
            6. Save the voucher
            
            Common Voucher Types:
            - F8: Sales Voucher
            - F9: Purchase Voucher
            - F5: Payment Voucher
            - F6: Receipt Voucher
            """
            chosen_link = "https://help.tallysolutions.com/tally-prime/accounting/voucher-entry/#sales-voucher"
            steps = [
                "Go to Gateway of Tally",
                "Press appropriate voucher key (F8/F9/F5/F6)",
                "Enter party name",
                "Enter item details",
                "Enter quantities and rates",
                "Save the voucher"
            ]
            
        else:
            context = """
            **General Tally Support**
            
            For general Tally Prime support:
            1. Go to Gateway of Tally
            2. Navigate to the appropriate menu
            3. Select the required option
            4. Enter the necessary details
            5. Save and verify the entry
            
            Common Menu Shortcuts:
            - Alt+G: GST Reports
            - Alt+M: Masters
            - Alt+I: Inventory Reports
            - F8: Sales Voucher
            - F9: Purchase Voucher
            """
            chosen_link = "https://help.tallysolutions.com/tally-prime/"
            steps = [
                "Go to Gateway of Tally",
                "Navigate to appropriate menu",
                "Select required option",
                "Enter necessary details",
                "Save and verify entry"
            ]
        
        default_links = [
            "https://help.tallysolutions.com/tally-prime/gst-reports/india-gst-faq-tally/#1362605",
            "https://help.tallysolutions.com/tally-prime/accounting/ledger-creation/#create-ledger",
            "https://help.tallysolutions.com/tally-prime/inventory/inventory-reports/#stock-summary",
            "https://help.tallysolutions.com/tally-prime/accounting/voucher-entry/#sales-voucher"
        ]
        
        return {
            "context": context.strip(),
            "raw_steps": steps,
            "candidate_links": default_links,
            "chosen_link": chosen_link,
            "hop1_urls": default_links
        }

# Global instance
knowledge_retriever = TallyKnowledgeRetriever()

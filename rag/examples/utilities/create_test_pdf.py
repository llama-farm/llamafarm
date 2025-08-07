#!/usr/bin/env python3
"""Create a sample PDF for testing the PDF parser."""

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from pathlib import Path

def create_test_pdf():
    """Create a test PDF with text, metadata, and structure."""
    pdf_path = Path("samples/test_document.pdf")
    pdf_path.parent.mkdir(exist_ok=True)
    
    # Create PDF with reportlab if available, otherwise create a simple text-based approach
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        width, height = letter
        
        # Set document metadata
        c.setTitle("Test RAG Document")
        c.setAuthor("RAG System")
        c.setSubject("Testing PDF parsing capabilities")
        c.setCreator("Test Script")
        
        # Page 1
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "Test Document for RAG System")
        
        c.setFont("Helvetica", 12)
        y_position = height - 100
        
        text_lines = [
            "This is a test document to demonstrate PDF parsing capabilities.",
            "",
            "The RAG system should be able to extract:",
            "• Text content from multiple pages",
            "• Document metadata (title, author, creation date)",
            "• Page structure and numbering",
            "",
            "Customer Support Scenario:",
            "A user reported login issues with the system. The problem occurred",
            "when trying to access the dashboard after a recent security update.",
            "The user received an error message: 'Authentication failed'.",
            "",
            "Resolution steps:",
            "1. Clear browser cache and cookies",
            "2. Reset password using the forgot password link",
            "3. Try logging in with a different browser",
            "4. Contact IT support if the issue persists",
        ]
        
        for line in text_lines:
            c.drawString(50, y_position, line)
            y_position -= 20
            if y_position < 100:  # Start new page
                c.showPage()
                c.setFont("Helvetica", 12)
                y_position = height - 50
        
        # Page 2
        c.showPage()
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 50, "Technical Documentation")
        
        c.setFont("Helvetica", 12)
        y_position = height - 100
        
        tech_content = [
            "System Requirements:",
            "• Python 3.9 or higher",
            "• ChromaDB for vector storage",
            "• Ollama for local embeddings",
            "",
            "Configuration:",
            "The system uses JSON configuration files to define:",
            "- Parser settings (CSV, PDF)",
            "- Embedder configuration (model, batch size)",
            "- Vector store settings (collection name, persistence)",
            "",
            "API Endpoints:",
            "• /ingest - Process and store documents",
            "• /search - Query the knowledge base",
            "• /info - Get system status",
            "",
            "This document contains multiple paragraphs and structured",
            "information that should be properly extracted by the PDF parser.",
        ]
        
        for line in tech_content:
            c.drawString(50, y_position, line)
            y_position -= 20
        
        c.save()
        print(f"✅ Created test PDF: {pdf_path}")
        return str(pdf_path)
        
    except ImportError:
        # Fallback: Create a simple text file and tell user to convert it
        text_content = """Test Document for RAG System

This is a test document to demonstrate PDF parsing capabilities.

The RAG system should be able to extract:
• Text content from multiple pages
• Document metadata (title, author, creation date)  
• Page structure and numbering

Customer Support Scenario:
A user reported login issues with the system. The problem occurred
when trying to access the dashboard after a recent security update.
The user received an error message: 'Authentication failed'.

Resolution steps:
1. Clear browser cache and cookies
2. Reset password using the forgot password link
3. Try logging in with a different browser
4. Contact IT support if the issue persists

Technical Documentation

System Requirements:
• Python 3.9 or higher
• ChromaDB for vector storage
• Ollama for local embeddings

Configuration:
The system uses JSON configuration files to define:
- Parser settings (CSV, PDF)
- Embedder configuration (model, batch size)
- Vector store settings (collection name, persistence)

API Endpoints:
• /ingest - Process and store documents
• /search - Query the knowledge base
• /info - Get system status

This document contains multiple paragraphs and structured
information that should be properly extracted by the PDF parser.
"""
        
        text_path = pdf_path.with_suffix('.txt')
        text_path.write_text(text_content)
        print(f"⚠️  Could not create PDF (reportlab not available)")
        print(f"📝 Created text file instead: {text_path}")
        print("💡 To create a PDF, install reportlab: pip install reportlab")
        return None

if __name__ == "__main__":
    create_test_pdf()
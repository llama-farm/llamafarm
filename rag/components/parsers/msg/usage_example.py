"""
Example usage of the MSG parser in the RAG system.

This demonstrates how to use the MSG parser that has been integrated
into the parser factory system for processing .msg files.
"""

# Example 1: Using the parser factory directly
from components.parsers import ParserFactory

# Create MSG parser by name
msg_parser = ParserFactory.create_parser("msg")

# Parse an MSG file
result = msg_parser.parse("email.msg")
documents = result.documents
errors = result.errors

# Example 2: Using the tool-aware factory
from components.parsers.parser_factory import ToolAwareParserFactory

# Create parser by file extension
msg_parser = ToolAwareParserFactory.get_parser_for_file("email.msg")

# Create parser with specific configuration
config = {
    "chunk_size": 500,
    "chunk_strategy": "email_sections",
    "extract_attachments": True,
    "include_attachment_content": True,
}
msg_parser = ToolAwareParserFactory.create_parser(
    parser_name="MsgParser_ExtractMsg", config=config
)

# Example 3: Using legacy aliases
from components.parsers import MsgParser

msg_parser = MsgParser()
result = msg_parser.parse("email.msg")

# Example 4: Parse MSG from bytes
with open("email.msg", "rb") as f:
    msg_data = f.read()

documents = msg_parser.parse_blob(msg_data, {"filename": "email.msg"})

# The parser will extract:
# - Email headers (From, To, Subject, Date, etc.)
# - Email body (both HTML and text)
# - Attachment metadata and optionally content
# - Rich metadata (sender, recipients, dates, message ID, etc.)
#
# Chunking strategies available:
# - "email_sections": Chunks by email components (headers, body, attachments)
# - "sentences": Uses LlamaIndex sentence splitter
# - "paragraphs": Splits by paragraph boundaries
# - "characters": Token-based chunking

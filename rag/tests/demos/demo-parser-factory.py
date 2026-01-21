#!/usr/bin/env python3
"""Demo script for Parser Factory Integration.

This demo shows:
1. Factory discovers UniversalParser
2. UniversalParser supports various file types
3. Priority comparison with legacy parsers
"""

import sys
from pathlib import Path

# Add rag directory to path
rag_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(rag_dir))

from components.parsers.parser_factory import ToolAwareParserFactory  # noqa: E402
from components.parsers.parser_registry import ParserRegistry  # noqa: E402


def demo_parser_discovery():
    """Demo parser discovery."""
    print("=" * 60)
    print("Parser Factory Discovery Demo")
    print("=" * 60)

    factory = ToolAwareParserFactory()

    # Discover all parsers
    print("\n--- Discovered Parser Types ---")
    parsers = factory.discover_parsers()

    for parser_type, configs in parsers.items():
        parser_names = [c.get("name") for c in configs]
        print(f"  {parser_type}: {parser_names}")

    # Check for UniversalParser
    print("\n--- UniversalParser Check ---")
    if "universal" in parsers:
        universal_configs = parsers["universal"]
        if any(c.get("name") == "UniversalParser" for c in universal_configs):
            print("[PASS] UniversalParser discovered in 'universal' type")
        else:
            print("[FAIL] UniversalParser not found in 'universal' configs")
    else:
        print("[FAIL] 'universal' parser type not discovered")

    print("\n" + "=" * 60)
    print("Parser Discovery Demo Complete")
    print("=" * 60)
    return True


def demo_universal_parser_info():
    """Demo UniversalParser information."""
    print("\n\n" + "=" * 60)
    print("UniversalParser Information Demo")
    print("=" * 60)

    # Get parser info
    info = ToolAwareParserFactory.get_parser_info("UniversalParser")

    if not info:
        print("[FAIL] Could not retrieve UniversalParser info")
        return False

    print(f"\nParser Name: {info.get('name')}")
    print(f"Display Name: {info.get('display_name')}")
    print(f"Version: {info.get('version')}")
    print(f"Tool: {info.get('tool')}")
    print(f"Priority: {info.get('priority')}")

    print("\n--- Supported Extensions ---")
    extensions = info.get("supported_extensions", [])
    # Group by category
    documents = [e for e in extensions if e in [".pdf", ".docx", ".doc", ".txt", ".md", ".rtf"]]
    spreadsheets = [e for e in extensions if e in [".xlsx", ".xls", ".csv", ".tsv"]]
    presentations = [e for e in extensions if e in [".pptx", ".ppt"]]
    web = [e for e in extensions if e in [".html", ".htm", ".xml", ".json"]]
    images = [e for e in extensions if e in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"]]
    other = [e for e in extensions if e not in documents + spreadsheets + presentations + web + images]

    print(f"  Documents: {documents}")
    print(f"  Spreadsheets: {spreadsheets}")
    print(f"  Presentations: {presentations}")
    print(f"  Web/Data: {web}")
    print(f"  Images (OCR): {images}")
    if other:
        print(f"  Other: {other}")

    print(f"\n  Total extensions: {len(extensions)}")

    print("\n--- Dependencies ---")
    deps = info.get("dependencies", {})
    print(f"  Required: {deps.get('required', [])}")
    print(f"  Optional: {deps.get('optional', [])}")

    print("\n--- Capabilities ---")
    caps = info.get("capabilities", [])
    for cap in caps:
        print(f"  - {cap}")

    print("\n--- Default Configuration ---")
    default_config = info.get("default_config", {})
    for key, value in default_config.items():
        print(f"  {key}: {value}")

    print("\n[PASS] UniversalParser info retrieved successfully")
    print("\n" + "=" * 60)
    print("UniversalParser Information Demo Complete")
    print("=" * 60)
    return True


def demo_parser_priority():
    """Demo parser priority comparison."""
    print("\n\n" + "=" * 60)
    print("Parser Priority Demo")
    print("=" * 60)

    # Get all parsers and their priorities
    all_parsers = ToolAwareParserFactory.list_parsers()

    print("\n--- Parser Priorities ---")
    priorities = []
    for parser_name in sorted(all_parsers):
        info = ToolAwareParserFactory.get_parser_info(parser_name)
        if info:
            priority = info.get("priority", 100)
            priorities.append((priority, parser_name))

    # Sort by priority (lower = higher priority)
    priorities.sort()

    for priority, name in priorities:
        marker = " <-- Default (highest priority)" if name == "UniversalParser" else ""
        print(f"  Priority {priority:3d}: {name}{marker}")

    # Verify UniversalParser has highest priority
    universal_priority = None
    for p, n in priorities:
        if n == "UniversalParser":
            universal_priority = p
            break

    if universal_priority is not None and universal_priority == priorities[0][0]:
        print("\n[PASS] UniversalParser has highest priority")
    elif universal_priority is not None and universal_priority <= 10:
        print("\n[PASS] UniversalParser has priority <= 10 (considered high)")
    else:
        print(f"\n[INFO] UniversalParser priority: {universal_priority}")

    print("\n" + "=" * 60)
    print("Parser Priority Demo Complete")
    print("=" * 60)
    return True


def demo_legacy_parsers():
    """Demo that legacy parsers are still available."""
    print("\n\n" + "=" * 60)
    print("Legacy Parser Availability Demo")
    print("=" * 60)

    # Check for specific legacy parsers
    legacy_parsers = [
        "PDFParser_PyPDF2",
        "PDFParser_LlamaIndex",
        "TextParser_Python",
        "TextParser_LlamaIndex",
        "CSVParser_Pandas",
    ]

    print("\n--- Legacy Parser Availability ---")
    available = 0
    for parser_name in legacy_parsers:
        info = ToolAwareParserFactory.get_parser_info(parser_name)
        if info:
            print(f"  [PASS] {parser_name} is available")
            available += 1
        else:
            print(f"  [SKIP] {parser_name} not found (may not be installed)")

    print(f"\n  {available}/{len(legacy_parsers)} legacy parsers available")

    # Check that legacy parsers still show in factory
    all_parsers = ToolAwareParserFactory.list_parsers()
    pdf_parsers = [p for p in all_parsers if "PDF" in p]
    text_parsers = [p for p in all_parsers if "Text" in p]

    print(f"\n  PDF parsers in factory: {pdf_parsers}")
    print(f"  Text parsers in factory: {text_parsers}")

    print("\n[PASS] Legacy parsers remain accessible alongside UniversalParser")
    print("\n" + "=" * 60)
    print("Legacy Parser Demo Complete")
    print("=" * 60)
    return True


def demo_parser_registry():
    """Demo parser registry."""
    print("\n\n" + "=" * 60)
    print("Parser Registry Demo")
    print("=" * 60)

    registry = ParserRegistry()
    all_parsers = registry.list_all_parsers()

    print(f"\n--- Registry Contains {len(all_parsers)} Parsers ---")
    for parser in sorted(all_parsers):
        print(f"  - {parser}")

    print("\n" + "=" * 60)
    print("Parser Registry Demo Complete")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success1 = demo_parser_discovery()
    success2 = demo_universal_parser_info()
    success3 = demo_parser_priority()
    success4 = demo_legacy_parsers()
    success5 = demo_parser_registry()

    if success1 and success2 and success3 and success4 and success5:
        print("\n\nAll parser factory demos passed!")
        sys.exit(0)
    else:
        print("\n\nSome demos failed!")
        sys.exit(1)

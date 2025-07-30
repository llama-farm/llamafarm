#!/bin/bash

# Quick Extractor Demo Script
# Tests all extractors with sample data without full system setup

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo -e "\n${CYAN}============================================================${NC}"
    echo -e "${CYAN}                    $1${NC}"
    echo -e "${CYAN}============================================================${NC}\n"
}

print_step() {
    echo -e "\n${BLUE}🔵 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Sample texts for testing
SAMPLE_TEXTS=(
    "Machine learning and artificial intelligence are transforming modern technology. Companies like OpenAI, Google, and Microsoft are leading innovations in natural language processing. The deadline for the project is January 15, 2024, and we need to complete the analysis by then. Contact john.doe@company.com for more information or call (555) 123-4567. The quarterly revenue increased by 25% to $2.5 million, which exceeded our expectations."
    
    "URGENT: Security breach detected at 2:30 PM on March 15, 2024. Unauthorized access attempt from IP 192.168.1.100. All users must change passwords immediately. Contact IT Security at security@company.org or call emergency hotline (800) 555-HELP. Estimated damage: $50,000. This incident requires immediate attention from Sarah Johnson, CISO, and the security team."
    
    "Legal Notice: The contract between ABC Corporation and XYZ Industries, signed on December 1, 2023, is hereby terminated effective February 29, 2024. All outstanding payments of $125,750.00 must be settled within 30 days. For questions, contact Legal Department at legal@abccorp.com or call (212) 555-0199. This document supersedes all previous agreements between the parties."
    
    "Research Report: Our analysis of 1,247 customer support tickets reveals that 34% are related to login issues, 28% to billing problems, and 38% to technical difficulties. Average resolution time has improved by 15% this quarter. The team processed tickets worth approximately $3.2 million in potential revenue impact. Key findings suggest implementing automated password reset and enhanced user documentation."
)

main() {
    print_header "🦙 Quick Extractor Demo 🦙"
    
    echo -e "${YELLOW}This demo tests all extractors with sample data${NC}"
    echo -e "${YELLOW}No full system setup required - just testing extraction capabilities${NC}\n"
    
    # Check if we're in the right directory
    if [[ ! -f "cli.py" ]]; then
        echo "❌ Please run this script from the RAG project directory"
        exit 1
    fi
    
    # Check if uv is available
    if ! command -v uv >/dev/null 2>&1; then
        echo "❌ uv not found. Please install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    
    print_step "Testing all extractors with different sample texts..."
    
    # Test each extractor with different samples
    extractors=("yake" "rake" "entities" "datetime" "statistics" "tfidf")
    
    for i in "${!extractors[@]}"; do
        extractor="${extractors[$i]}"
        sample_index=$((i % ${#SAMPLE_TEXTS[@]}))
        sample_text="${SAMPLE_TEXTS[$sample_index]}"
        
        print_step "Testing $extractor extractor..."
        echo -e "${YELLOW}Sample text: ${sample_text:0:100}...${NC}\n"
        
        if uv run python cli.py extractors test --extractor "$extractor" --text "$sample_text"; then
            print_success "$extractor extractor test completed"
        else
            echo "⚠️  $extractor extractor test had issues"
        fi
        
        echo -e "\n${CYAN}Press Enter to continue to next extractor...${NC}"
        read -r
    done
    
    print_header "🎯 Extractor Comparison Demo"
    
    echo -e "${YELLOW}Now let's compare all keyword extractors on the same text:${NC}\n"
    
    comparison_text="Artificial intelligence and machine learning are revolutionizing healthcare, finance, and education. Natural language processing enables computers to understand human communication. Deep learning models require significant computational resources and large datasets for training. The future of AI depends on ethical considerations, regulatory frameworks, and responsible development practices."
    
    echo -e "${CYAN}Sample text for comparison:${NC}"
    echo -e "${YELLOW}$comparison_text${NC}\n"
    
    keyword_extractors=("yake" "rake" "tfidf")
    
    for extractor in "${keyword_extractors[@]}"; do
        print_step "Running $extractor on comparison text..."
        uv run python cli.py extractors test --extractor "$extractor" --text "$comparison_text"
        echo -e "\n${CYAN}Press Enter for next comparison...${NC}"
        read -r
    done
    
    print_header "📊 Statistics and Analysis Demo"
    
    print_step "Running comprehensive content analysis..."
    
    analysis_text="The quick brown fox jumps over the lazy dog. This pangram contains every letter of the English alphabet at least once. It's commonly used for testing typewriters, computer keyboards, and fonts. The sentence has been around since the late 1800s and remains popular today. Modern variations include 'Pack my box with five dozen liquor jugs' and 'Waltz, bad nymph, for quick jigs vex.' These sentences serve as excellent examples for typography and linguistic analysis."
    
    echo -e "${YELLOW}Analyzing text for readability, structure, and content statistics...${NC}\n"
    uv run python cli.py extractors test --extractor "statistics" --text "$analysis_text"
    
    print_header "🎉 Quick Demo Complete!"
    
    echo -e "${GREEN}✅ All extractors tested successfully!${NC}\n"
    
    echo -e "${CYAN}Key Takeaways:${NC}"
    echo -e "${YELLOW}• YAKE: Best for single documents, considers word position${NC}"
    echo -e "${YELLOW}• RAKE: Fast phrase extraction, good for technical text${NC}"
    echo -e "${YELLOW}• TF-IDF: Best when you have multiple documents to compare${NC}"
    echo -e "${YELLOW}• Entities: Finds people, places, dates, emails, phones, money${NC}"
    echo -e "${YELLOW}• DateTime: Extracts dates, times, and relative expressions${NC}"
    echo -e "${YELLOW}• Statistics: Comprehensive text analysis and readability${NC}\n"
    
    echo -e "${CYAN}Next Steps:${NC}"
    echo -e "${YELLOW}• Run the full setup: ./setup_and_demo.sh${NC}"
    echo -e "${YELLOW}• Try with your own files: uv run python cli.py extractors test --extractor yake --file yourfile.txt${NC}"
    echo -e "${YELLOW}• Check available extractors: uv run python cli.py extractors list --detailed${NC}"
    echo -e "${YELLOW}• See configuration examples in config_examples/extractors_demo_config.json${NC}"
    
    echo -e "\n${CYAN}🦙 No prob-llama with these extractors! 🦙${NC}"
}

main "$@"
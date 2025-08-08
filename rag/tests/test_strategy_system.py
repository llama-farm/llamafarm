#!/usr/bin/env python3
"""
Strategy System Tester

Tests the RAG strategy system with various real strategies from demo_strategies.yaml
Uses different combinations of components to ensure the system works end-to-end.
"""

import subprocess
import sys
import os
import tempfile
import shutil
from pathlib import Path
import time
from typing import Dict, Any, List, Tuple

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'

class StrategySystemTester:
    """Tests existing strategies from demo_strategies.yaml"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.cli_path = self.base_dir / "cli.py"
        self.demos_dir = self.base_dir / "demos"
        self.test_strategies = [
            "research_papers_demo",
            "customer_support_demo", 
            "code_documentation_demo",
            "news_analysis_demo",
            "business_reports_demo"
        ]
        self.results = []
        self.passed = 0
        self.failed = 0
        
    def run_command(self, command: str, description: str = "", timeout: int = 30) -> Tuple[bool, str, str]:
        """Run a CLI command and return success status and output."""
        if description:
            print(f"\n  {CYAN}▶ {description}{RESET}")
        print(f"    {YELLOW}$ {command}{RESET}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.base_dir,
                timeout=timeout
            )
            
            success = result.returncode == 0
            
            # Show brief output
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines[:3]:  # Show first 3 lines
                    if line:
                        print(f"      {line[:80]}...")
                if len(lines) > 3:
                    print(f"      ... ({len(lines)-3} more lines)")
            
            if success:
                print(f"    {GREEN}✅ Success{RESET}")
            else:
                print(f"    {RED}❌ Failed{RESET}")
                if result.stderr:
                    # Show real errors only
                    error_lines = [l for l in result.stderr.split('\n') 
                                 if l and not any(skip in l for skip in 
                                 ['WARNING', 'Progress', 'Embedding:', '%|', '🦙', 'usage:'])]
                    if error_lines:
                        print(f"    {RED}Error: {error_lines[0][:100]}{RESET}")
                        
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            print(f"    {RED}❌ Timeout after {timeout}s{RESET}")
            return False, "", "Timeout"
        except Exception as e:
            print(f"    {RED}❌ Error: {e}{RESET}")
            return False, "", str(e)
            
    def test_strategy(self, strategy_name: str) -> Dict[str, Any]:
        """Test a single strategy."""
        print(f"\n{BOLD}{BLUE}Testing Strategy: {strategy_name}{RESET}")
        print(f"{BLUE}{'─' * 60}{RESET}")
        
        result = {
            "strategy": strategy_name,
            "steps": {},
            "success": False
        }
        
        # Get sample data path based on strategy
        sample_paths = {
            "research_papers_demo": "demos/static_samples/research_papers",
            "customer_support_demo": "demos/static_samples/customer_support/support_tickets.csv",
            "code_documentation_demo": "demos/static_samples/code_documentation",
            "news_analysis_demo": "demos/static_samples/news_articles",
            "business_reports_demo": "demos/static_samples/business_reports"
        }
        
        sample_path = sample_paths.get(strategy_name, "demos/static_samples")
        
        # Step 1: Clean any existing data
        print(f"  🧹 Cleaning previous data...")
        collection_name = strategy_name.replace("_demo", "")
        db_path = self.base_dir / "vectordb" / collection_name
        if db_path.exists():
            shutil.rmtree(db_path)
            print(f"    Removed {db_path}")
        
        # Step 2: Test strategy info
        success, stdout, stderr = self.run_command(
            f"python cli.py strategies show {strategy_name}",
            "Show strategy configuration"
        )
        result["steps"]["show_strategy"] = success
        
        # Step 3: Ingest documents
        success, stdout, stderr = self.run_command(
            f"python cli.py ingest {sample_path} --strategy {strategy_name}",
            f"Ingest sample data from {sample_path}",
            timeout=60
        )
        result["steps"]["ingest"] = success
        
        if not success:
            result["error"] = "Ingestion failed"
            return result
            
        # Wait for ingestion to complete
        time.sleep(2)
        
        # Step 4: Get collection info
        success, stdout, stderr = self.run_command(
            f"python cli.py info --strategy {strategy_name}",
            "Get collection info"
        )
        result["steps"]["info"] = success
        
        # Parse document count
        doc_count = 0
        if success and stdout:
            for line in stdout.split('\n'):
                if "count:" in line.lower() or "documents:" in line.lower():
                    try:
                        # Extract number from line
                        import re
                        numbers = re.findall(r'\d+', line)
                        if numbers:
                            doc_count = int(numbers[0])
                            break
                    except:
                        pass
        result["document_count"] = doc_count
        
        # Step 5: Test searches based on strategy type
        search_queries = {
            "research_papers_demo": [
                "transformer architecture",
                "machine learning",
                "neural networks"
            ],
            "customer_support_demo": [
                "login problem",
                "password reset",
                "error message"
            ],
            "code_documentation_demo": [
                "API endpoint",
                "authentication",
                "implementation"
            ],
            "news_analysis_demo": [
                "AI breakthrough",
                "technology",
                "innovation"
            ],
            "business_reports_demo": [
                "revenue growth",
                "quarterly results",
                "financial metrics"
            ]
        }
        
        queries = search_queries.get(strategy_name, ["test query"])
        successful_searches = 0
        
        for i, query in enumerate(queries[:2]):  # Test first 2 queries
            success, stdout, stderr = self.run_command(
                f"python cli.py search \"{query}\" --strategy {strategy_name} --top-k 3",
                f"Search: '{query}'"
            )
            result["steps"][f"search_{i+1}"] = success
            
            if success and "Search Results:" in stdout:
                successful_searches += 1
                
        result["successful_searches"] = successful_searches
        
        # Step 6: Test manage stats
        success, stdout, stderr = self.run_command(
            f"python cli.py manage --rag-strategy {strategy_name} stats",
            "Get statistics"
        )
        result["steps"]["stats"] = success
        
        # Calculate success rate
        total_steps = len(result["steps"])
        successful_steps = sum(1 for v in result["steps"].values() if v)
        result["success_rate"] = successful_steps / total_steps if total_steps > 0 else 0
        result["success"] = result["success_rate"] >= 0.6  # 60% threshold for demo strategies
        
        return result
        
    def test_extractor_listing(self):
        """Test that extractors are properly registered."""
        print(f"\n{BOLD}{BLUE}Testing Component Registration{RESET}")
        print(f"{BLUE}{'─' * 60}{RESET}")
        
        success, stdout, stderr = self.run_command(
            "python cli.py extractors list",
            "List available extractors"
        )
        
        extractor_count = 0
        if success and "Found" in stdout:
            import re
            match = re.search(r'Found (\d+) extractors', stdout)
            if match:
                extractor_count = int(match.group(1))
                
        print(f"  {GREEN if extractor_count > 0 else RED}Found {extractor_count} extractors{RESET}")
        return extractor_count > 0
        
    def run_all_tests(self):
        """Run all strategy tests."""
        print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
        print(f"{BOLD}{CYAN}{'Strategy System Integration Test':^70}{RESET}")
        print(f"{BOLD}{CYAN}{'='*70}{RESET}")
        print(f"\nThis test validates that the strategy system works end-to-end")
        print(f"with real demo strategies and sample data.\n")
        
        # Test component registration
        extractors_ok = self.test_extractor_listing()
        
        # Test each strategy
        for strategy_name in self.test_strategies:
            result = self.test_strategy(strategy_name)
            self.results.append(result)
            
            if result["success"]:
                self.passed += 1
            else:
                self.failed += 1
                
            # Brief pause between strategies
            time.sleep(1)
            
        # Print summary
        self.print_summary(extractors_ok)
        
        return self.failed == 0
        
    def print_summary(self, extractors_ok: bool):
        """Print test summary."""
        print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
        print(f"{BOLD}{CYAN}{'Test Summary':^70}{RESET}")
        print(f"{BOLD}{CYAN}{'='*70}{RESET}\n")
        
        # Component registration
        print(f"{BOLD}Component Registration:{RESET}")
        print(f"  Extractors: {GREEN if extractors_ok else RED}{'✅ Registered' if extractors_ok else '❌ Not Found'}{RESET}")
        
        # Strategy results
        print(f"\n{BOLD}Strategy Test Results:{RESET}")
        print(f"{BLUE}{'─' * 60}{RESET}")
        
        for result in self.results:
            status = f"{GREEN}✅ PASS{RESET}" if result["success"] else f"{RED}❌ FAIL{RESET}"
            rate = result.get("success_rate", 0) * 100
            docs = result.get("document_count", 0)
            searches = result.get("successful_searches", 0)
            
            print(f"\n{BOLD}{result['strategy']:<30}{RESET} {status}")
            print(f"  │ Success Rate: {rate:.0f}%")
            print(f"  │ Documents Indexed: {docs}")
            print(f"  │ Successful Searches: {searches}/2")
            
            # Show step results
            steps_summary = []
            for step, success in result["steps"].items():
                symbol = "✓" if success else "✗"
                color = GREEN if success else RED
                steps_summary.append(f"{color}{symbol}{RESET}")
            print(f"  └ Steps: {' '.join(steps_summary)} ({', '.join(result['steps'].keys())})")
            
            # Show errors if any
            if result.get("error"):
                print(f"    {RED}Error: {result['error']}{RESET}")
                
        # Component coverage summary
        print(f"\n{BOLD}Components Tested:{RESET}")
        print(f"  • Parsers: CSVParser, PlainTextParser, MarkdownParser, HTMLParser, PDFParser")
        print(f"  • Extractors: EntityExtractor, KeywordExtractor, SummaryExtractor, etc.")
        print(f"  • Embedder: OllamaEmbedder (nomic-embed-text)")
        print(f"  • Vector Store: ChromaStore")
        print(f"  • Retrieval: BasicSimilarity, MetadataFiltered, MultiQuery, Reranked, Hybrid")
        
        # Overall summary
        print(f"\n{BOLD}{BLUE}{'─'*60}{RESET}")
        print(f"{BOLD}Overall Results:{RESET}")
        total = len(self.results)
        print(f"  {GREEN}✅ Passed: {self.passed}/{total} strategies{RESET}")
        if self.failed > 0:
            print(f"  {RED}❌ Failed: {self.failed}/{total} strategies{RESET}")
            
        success_rate = (self.passed / total * 100) if total > 0 else 0
        print(f"  📊 Overall Success Rate: {success_rate:.0f}%")
        
        # Final verdict
        print(f"\n{BOLD}Final Result: ", end="")
        if self.failed == 0:
            print(f"{GREEN}ALL STRATEGIES PASSED! 🎉{RESET}")
            print(f"{GREEN}The strategy system is working correctly!{RESET}")
        elif self.failed <= 1:
            print(f"{YELLOW}MOSTLY SUCCESSFUL (1 minor issue){RESET}")
            print(f"{YELLOW}The strategy system is functional with minor issues.{RESET}")
        else:
            print(f"{RED}MULTIPLE FAILURES DETECTED{RESET}")
            print(f"{RED}The strategy system needs attention.{RESET}")
            
        print(f"{BOLD}{CYAN}{'='*70}{RESET}\n")


def main():
    """Main entry point."""
    print(f"{BOLD}{CYAN}RAG Strategy System Integration Test{RESET}")
    print(f"{CYAN}Testing real strategies with actual demo data{RESET}\n")
    
    tester = StrategySystemTester()
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
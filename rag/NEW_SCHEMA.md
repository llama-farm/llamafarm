# RAG Subsystem New Schema Migration Plan

## Overview

This document outlines the complete migration plan for aligning the RAG subsystem with the new schema architecture. The migration moves from a monolithic "strategy-based" approach to a clean separation of "databases" and "data_processing_strategies" with dedicated CLI arguments.

## Current vs New Architecture

### Current (Old Strategy System)
- **Single Strategy**: One combined configuration with all components bundled together
- **CLI Arguments**: `--strategy <strategy_name>` for everything
- **Schema**: Strategies contain parser, embedder, vector_store, retrieval_strategy, extractors in one object
- **Files**: `default_strategies.yaml` with monolithic strategy definitions
- **Components**: StrategyManager, StrategyLoader, StrategyConfig classes
- **Config Support**: Limited integration with global `llamafarm.yaml` (only in API)

### New (Databases + Data Processing Strategies + Global Config)
- **Separated Architecture**: 
  - `databases` - Vector stores with their own embedding/retrieval strategies
  - `data_processing_strategies` - File processing pipelines (parsers + extractors)
- **CLI Arguments**: 
  - `--config <llamafarm.yaml|llamafarm.toml>` for global project configuration
  - `--database <db_name>` (required) for ingestion and querying
  - `--data-processing-strategy <strategy_name>` (required) for ingestion
  - `--retrieval-strategy <strategy_name>` (optional) for querying
- **Schema**: 
  - **Global**: Uses `/config/schema.yaml` (LlamaFarm project schema)
  - **RAG**: Uses `/rag/schema.yaml` embedded via `$ref: "../rag/schema.yaml"`
- **Files**: 
  - **Only**: `llamafarm.yaml` or `llamafarm.toml` (global project config)
- **Components**: Enhanced SchemaHandler + Global ConfigLoader integration

## Migration Requirements

### ✅ Completed Research
1. **Current Schema Understanding**: Old strategy-based system analyzed
2. **New Schema Understanding**: Databases + data_processing_strategies architecture analyzed
3. **CLI Analysis**: Current arguments and required changes identified
4. **Dead Code Identification**: Legacy components that need removal identified

### 🎯 Required Changes

## Phase 1: CLI Command Structure Overhaul

### 1.1 Remove Combined Strategy Arguments
**Files to modify**: `rag/cli.py`

**Current problematic patterns**:
```bash
# OLD - Combined strategy approach
python cli.py ingest data.csv --strategy simple
python cli.py search "query" --strategy customer_support
```

**New optimal patterns**:
```bash
# NEW - Global config with separate database and processing strategy
python cli.py --config llamafarm.yaml ingest data.csv --database main_database --data-processing-strategy text_processing
python cli.py --config llamafarm.yaml search "query" --database main_database --retrieval-strategy basic_search
```

### 1.2 CLI Argument Changes

#### Global Configuration Arguments
- **Enhance**: `--config <path>` to support both `llamafarm.yaml` and `llamafarm.toml` files
- **Add**: Auto-detection of global config files (`llamafarm.yaml`, `llamafarm.toml`) in current directory
- **Add**: Schema validation using `/config/schema.yaml` for global configs

#### Ingestion Command (`ingest`)
- **Remove**: `--strategy` argument
- **Add**: `--database <database_name>` (required)
- **Add**: `--data-processing-strategy <strategy_name>` (required)
- **Keep**: All existing parser/embedder overrides for compatibility

#### Query Command (`search`)  
- **Remove**: `--strategy` argument
- **Add**: `--database <database_name>` (required)
- **Add**: `--retrieval-strategy <strategy_name>` (optional, uses database default if not specified)
- **Keep**: All existing retrieval overrides for compatibility

#### Info Command (`info`)
- **Remove**: `--strategy` argument  
- **Add**: `--database <database_name>` (optional, shows all databases if not specified)

#### Management Commands (`manage`)
- **Remove**: `--strategy` / `--rag-strategy` arguments
- **Add**: `--database <database_name>` (required for database-specific operations)

### 1.3 Error Handling & User Experience
- Clear error messages when old `--strategy` arguments are used
- Helpful suggestions to migrate to new argument structure
- List available databases and data processing strategies when arguments are missing

## Phase 2: Remove Legacy Strategy System

### 2.1 Delete Dead Code Files
**Files to remove**:
```
rag/core/strategies/manager.py
rag/core/strategies/manager.py.deprecated  
rag/core/strategies/config.py
rag/core/strategies/config.py.deprecated
rag/core/strategies/loader.py  # Only if not needed by SchemaHandler
rag/default_strategies.yaml    # Old format strategies
```

### 2.2 Remove Legacy Imports
**Files to clean up**:
- `rag/cli.py`: Remove StrategyManager imports (line 38 already commented)
- All test files: Update imports to use SchemaHandler instead of StrategyManager
- Update any remaining references to old strategy classes

### 2.3 Update Core Components
**Files to modify**:
- `rag/core/strategies/__init__.py`: Remove exports of deleted classes
- Update factory methods in `rag/core/factories.py` if they depend on legacy strategy system

## Phase 3: Global Config Integration & Schema Handler Enhancement

### 3.1 Integrate Global ConfigLoader
**Files**: `rag/cli.py`, `rag/core/strategies/handler.py`

**Add Global Config Support**:
```python
from config.helpers.loader import ConfigLoader

def load_global_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load global LlamaFarm configuration with auto-detection."""
    if config_path:
        # Explicit config file provided
        return ConfigLoader.from_file(config_path)
    
    # Auto-detect in current directory
    for filename in ['llamafarm.yaml', 'llamafarm.toml']:
        if Path(filename).exists():
            return ConfigLoader.from_file(filename)
    
    # No config found
    raise ValueError("No global config found. Create llamafarm.yaml or llamafarm.toml")

def get_rag_config_from_global(global_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract RAG configuration from global config."""
    return global_config.get('rag', {})

```

### 3.2 Enhance SchemaHandler for Global Config
**File**: `rag/core/strategies/handler.py`

**Add global config constructor and methods**:
```python
class SchemaHandler:
    def __init__(self, config_source: str):
        """Initialize with global LlamaFarm config file only."""
        self.config_source = Path(config_source)
        self.global_config = None
        self.rag_config = None
        
        if self.config_source.exists():
            # Load via global ConfigLoader with validation
            from config.helpers.loader import ConfigLoader
            self.global_config = ConfigLoader.from_file(str(config_source))
            self.rag_config = self.global_config.get('rag', {})
        else:
            raise ValueError(f"Global config file not found: {config_source}")
    
    # Methods for database and strategy access
    def get_database_config(self, database_name: str) -> Dict[str, Any]
    def get_data_processing_strategy_config(self, strategy_name: str) -> Dict[str, Any]  
    def get_database_embedding_strategy(self, database_name: str, strategy_name: Optional[str] = None) -> Dict[str, Any]
    def get_database_retrieval_strategy(self, database_name: str, strategy_name: Optional[str] = None) -> Dict[str, Any]
    def list_databases(self) -> List[str]
    def list_data_processing_strategies(self) -> List[str]
    def list_database_retrieval_strategies(self, database_name: str) -> List[str]
    def validate_database_name(self, database_name: str) -> bool
    def validate_data_processing_strategy_name(self, strategy_name: str) -> bool
```

### 3.3 CLI Integration Points
**File**: `rag/cli.py`

**Functions to modify**:
- `load_config_with_strategy()`: Replace with `load_global_config()`
- `handle_ingest_command()`: Use `--database` and `--data-processing-strategy`
- `handle_search_command()`: Use `--database` and optional `--retrieval-strategy`
- `handle_info_command()`: Use `--database`
- `handle_manage_command()`: Use `--database`
- All strategy resolution functions: Replace StrategyManager with enhanced SchemaHandler

**New config resolution logic**:
```python
def resolve_config_and_handler(args) -> Tuple[Dict[str, Any], SchemaHandler]:
    """Resolve configuration and create appropriate handler."""
    # Try global config first
    global_config = load_global_config(getattr(args, 'config', None))
    
    if global_config:
        # Global LlamaFarm config with schema validation
        handler = SchemaHandler(args.config or 'llamafarm.yaml')
        return global_config, handler
    else:
        raise ValueError("No global configuration found. Provide --config llamafarm.yaml")
```

## Phase 4: Update Configuration and Documentation

### 4.1 Configuration Migration
- **Remove**: `rag/default_strategies.yaml` (old format)
- **Remove**: All RAG-specific config files and examples
- **Create**: Global config templates in `/config/templates/`
- **Update**: Documentation to point to global config approach only

### 4.2 Example Command Updates
**Update examples throughout codebase**:
```bash
# OLD examples to replace
python cli.py ingest docs/ --strategy research

# NEW examples (global config only)
python cli.py --config llamafarm.yaml ingest docs/ --database research_papers_db --data-processing-strategy pdf_processing
```

### 4.3 Help Text and Documentation
- Update all CLI help text to show global config approach only
- Update examples in `rag/README.md` to use `llamafarm.yaml`
- Remove strategy documentation files (replaced by global config docs)
- Ensure error messages guide users to global config format

## Phase 5: Testing and Validation

### 5.1 Test Updates
**Files to modify**:
- All test files using old `--strategy` arguments
- Update integration tests to use new CLI structure
- Add tests for new argument validation
- Test error cases with helpful error messages

### 5.2 Demo Updates  
**Files to verify/update**:
- All demo files should already use new schema (confirmed in research)
- Ensure all example commands use new argument structure
- Update any remaining references to old strategy system

## Implementation Phases Summary

### Phase 1: CLI Overhaul (Priority: CRITICAL)
- **Time**: 1-2 days
- **Risk**: HIGH (breaks existing CLI usage)
- **Dependencies**: None
- **Deliverable**: New CLI argument structure working

### Phase 2: Dead Code Removal (Priority: HIGH)  
- **Time**: 1 day
- **Risk**: MEDIUM (import errors possible)
- **Dependencies**: Phase 1 complete
- **Deliverable**: Clean codebase without legacy strategy system

### Phase 3: Schema Handler Enhancement (Priority: HIGH)
- **Time**: 1-2 days  
- **Risk**: MEDIUM (integration complexity)
- **Dependencies**: Phase 2 complete
- **Deliverable**: Full new schema support in CLI

### Phase 4: Documentation Updates (Priority: MEDIUM)
- **Time**: 1 day
- **Risk**: LOW
- **Dependencies**: Phase 3 complete  
- **Deliverable**: Updated documentation and examples

### Phase 5: Testing & Validation (Priority: HIGH)
- **Time**: 1-2 days
- **Risk**: LOW
- **Dependencies**: Phases 1-4 complete
- **Deliverable**: Fully tested new system

## Breaking Changes Notice

### For Users
This migration introduces **BREAKING CHANGES** to the CLI interface:

**Before**:
```bash
python cli.py ingest data.csv --strategy simple
python cli.py search "query" --strategy simple  
```

**After (Global Config - Required)**:
```bash
# Using global config with explicit database/strategy  
python cli.py --config llamafarm.yaml ingest data.csv --database main_database --data-processing-strategy text_processing
python cli.py --config llamafarm.yaml search "query" --database main_database --retrieval-strategy basic_search
```


### Migration Path for Users
1. **Required**: Create `llamafarm.yaml` with global project configuration
   - Define RAG `databases` and `data_processing_strategies` sections
2. **Update all CLI commands**: Use new argument structure:
   - Always specify `--config llamafarm.yaml`
   - Always specify `--database <database_name>`
   - For ingestion: specify `--data-processing-strategy <strategy_name>`
   - For querying: optionally specify `--retrieval-strategy <strategy_name>`

## Success Criteria

### ✅ Migration Complete When:
1. **No legacy strategy arguments**: All `--strategy` arguments removed from CLI
2. **Global config integration**: Full support for `llamafarm.yaml` and `llamafarm.toml` files
3. **Schema validation**: Global configs validated against `/config/schema.yaml`
4. **New arguments working**: `--database`, `--data-processing-strategy`, `--retrieval-strategy` fully functional
6. **Auto-detection**: CLI automatically detects `llamafarm.yaml`/`llamafarm.toml` in current directory
7. **Clean codebase**: All legacy strategy system files and imports removed
8. **Global config only**: Only `llamafarm.yaml`/`llamafarm.toml` files supported
9. **Comprehensive testing**: All functionality working with global configs
10. **Updated documentation**: All examples prioritize global config approach
11. **User-friendly errors**: Clear guidance when users try old argument patterns

## Risk Mitigation

### Rollback Plan
- Keep legacy code in separate branch until migration fully validated
- Comprehensive test suite ensures no functionality regression
- Staged rollout possible if needed

### User Communication
- Clear breaking changes documentation
- Migration guide with before/after examples
- Helpful error messages during transition period

## Files Requiring Attention

### Critical Files (Must Modify)
- `rag/cli.py` - Primary CLI interface
- `rag/core/strategies/handler.py` - Schema handling
- All test files using `--strategy` arguments

### Files to Remove  
- `rag/core/strategies/manager.py`
- `rag/core/strategies/config.py`
- `rag/core/strategies/loader.py` (if not used by SchemaHandler)
- `rag/default_strategies.yaml` (old format)

### Documentation Files
- `rag/README.md`
- `rag/docs/*.md`
- All demo documentation
- CLI help text and examples

---

## Detailed Implementation Phases

### Phase 1: CLI Argument Structure Overhaul

#### Step 1.1: Update Argument Parsing (rag/cli.py)
```python
# Enhance global config argument:
parser.add_argument(
    "--config", "-c",
    help="Configuration file path (llamafarm.yaml, llamafarm.toml, or legacy RAG config)"
)

# Replace in ingest_parser.add_argument section:
- Remove: "--strategy", help="Strategy name to use instead of config file"
+ Add: "--database", required=True, help="Database name for ingestion"  
+ Add: "--data-processing-strategy", required=True, help="Data processing strategy name"

# Replace in search_parser.add_argument section:
- Remove: "--strategy", help="Strategy name to use instead of config file"
+ Add: "--database", required=True, help="Database name for search"
+ Add: "--retrieval-strategy", help="Retrieval strategy name (optional, uses database default)"

# Replace in info_parser.add_argument section:
- Remove: "--strategy", help="Use a predefined strategy for configuration"
+ Add: "--database", help="Database name for info (optional, shows all if not specified)"
```

#### Step 1.2: Update CLI Handler Functions
```python
# Update function signatures:
- def handle_ingest_command(args, strategy_name=None):
+ def handle_ingest_command(args, database_name, data_processing_strategy):

- def handle_search_command(args, strategy_name=None):  
+ def handle_search_command(args, database_name, retrieval_strategy=None):

- def handle_info_command(args, strategy_name=None):
+ def handle_info_command(args, database_name=None):

# Add global config resolution:
def resolve_command_config(args):
    """Resolve configuration for a command from global config."""
    global_config, handler = resolve_config_and_handler(args)
    
    return {
        'database': args.database,
        'data_processing_strategy': getattr(args, 'data_processing_strategy', None),
        'retrieval_strategy': getattr(args, 'retrieval_strategy', None)
    }, handler
```

#### Step 1.3: Add Validation and Error Handling
```python
def validate_new_cli_args(args):
    """Validate new CLI argument combinations and provide helpful errors."""
    if hasattr(args, 'strategy'):
        print("❌ Error: --strategy argument is no longer supported")
        print("💡 New approach:")
        print("   Use --config llamafarm.yaml --database <name> --data-processing-strategy <name>")
        print("📝 Examples:")
        print("   python cli.py --config llamafarm.yaml ingest data.csv --database main_db --data-processing-strategy text_processing")
        print("   python cli.py --config llamafarm.yaml search 'query' --database main_db")
        sys.exit(1)

def validate_command_args(args, command_type):
    """Validate command-specific argument combinations."""    
    if command_type == 'ingest':
        if not (hasattr(args, 'database') and args.database):
            print("❌ Error: --database argument is required for ingestion")
            print("📋 Available databases:")
            # Show available databases
            sys.exit(1)
            
        if not (hasattr(args, 'data_processing_strategy') and args.data_processing_strategy):
            print("❌ Error: --data-processing-strategy argument is required for ingestion")
            print("📋 Available strategies:")
            # Show available strategies
            sys.exit(1)
    
    elif command_type == 'search':
        if not (hasattr(args, 'database') and args.database):
            print("❌ Error: --database argument is required for search")
            print("📋 Available databases:")
            # Show available databases  
            sys.exit(1)
```

### Phase 2: Legacy Code Removal

#### Step 2.1: Remove Strategy Manager Files
```bash
rm rag/core/strategies/manager.py
rm rag/core/strategies/manager.py.deprecated
rm rag/core/strategies/config.py  
rm rag/core/strategies/config.py.deprecated
rm rag/default_strategies.yaml
# Remove all RAG-specific config examples
rm -rf rag/config_examples/
rm -rf rag/samples/*strategies.yaml
rm rag/demos/demo_strategies.yaml  # Replace with global config
```

#### Step 2.2: Clean Up Imports
```python
# In rag/cli.py - Remove these lines:
- from core.strategies import StrategyManager  # Already commented line 38
- Any remaining StrategyManager references

# In rag/core/strategies/__init__.py - Remove:
- from .manager import StrategyManager
- from .config import StrategyConfig  
- __all__ = [...] # Update to remove deleted classes
```

#### Step 2.3: Verify No Broken Dependencies
```bash
# Check for remaining references:
grep -r "StrategyManager" rag/
grep -r "StrategyConfig" rag/
grep -r "from.*strategies.*manager" rag/
```

### Phase 3: Schema Handler Enhancement

#### Step 3.1: Extend SchemaHandler Methods
Add to `rag/core/strategies/handler.py`:
```python
def get_database_names(self) -> List[str]:
    """Get list of available database names."""
    return [db['name'] for db in self.rag_config.get('databases', [])]

def get_data_processing_strategy_names(self) -> List[str]:
    """Get list of available data processing strategy names."""  
    return [strategy['name'] for strategy in self.rag_config.get('data_processing_strategies', [])]

def get_database_retrieval_strategies(self, database_name: str) -> List[str]:
    """Get available retrieval strategies for a database."""
    for db in self.rag_config.get('databases', []):
        if db['name'] == database_name:
            return [rs['name'] for rs in db.get('retrieval_strategies', [])]
    return []

def create_database_config(self, database_name: str) -> Dict[str, Any]:
    """Create database configuration for factories."""
    for db in self.rag_config.get('databases', []):
        if db['name'] == database_name:
            return {
                'vector_store': {
                    'type': db['type'],
                    'config': db.get('config', {})
                },
                'default_embedding_strategy': db.get('default_embedding_strategy'),
                'default_retrieval_strategy': db.get('default_retrieval_strategy'),
                'embedding_strategies': db.get('embedding_strategies', []),
                'retrieval_strategies': db.get('retrieval_strategies', [])
            }
    raise ValueError(f"Database '{database_name}' not found")

def create_processing_config(self, strategy_name: str) -> Dict[str, Any]:
    """Create data processing strategy configuration."""
    for strategy in self.rag_config.get('data_processing_strategies', []):
        if strategy['name'] == strategy_name:
            return {
                'parsers': strategy.get('parsers', []),
                'extractors': strategy.get('extractors', []),
                'directory_config': strategy.get('directory_config', {})
            }
    raise ValueError(f"Data processing strategy '{strategy_name}' not found")
```

#### Step 3.2: Update CLI Integration Points
Replace in `rag/cli.py`:
```python
# Replace StrategyManager usage:
- strategy_manager = StrategyManager(args.strategy_file)
- config = strategy_manager.convert_strategy_to_config(strategy_name)

+ schema_handler = SchemaHandler(args.strategy_file)  
+ db_config = schema_handler.create_database_config(database_name)
+ proc_config = schema_handler.create_processing_config(data_processing_strategy)
```

### Phase 4: Update All Commands

#### Step 4.1: Ingest Command Implementation
```python
def handle_ingest_command(args, database_name, data_processing_strategy):
    """Handle document ingestion with new schema."""
    # Create configurations
    schema_handler = SchemaHandler(args.config)
    db_config = schema_handler.create_database_config(database_name)
    proc_config = schema_handler.create_processing_config(data_processing_strategy)
    
    # Rest of ingestion logic...
```

#### Step 4.2: Search Command Implementation  
```python
def handle_search_command(args, database_name, retrieval_strategy=None):
    """Handle search with new schema."""
    schema_handler = SchemaHandler(args.config)
    db_config = schema_handler.create_database_config(database_name)
    
    # Use retrieval_strategy if specified, otherwise use database default
    if not retrieval_strategy:
        retrieval_strategy = db_config.get('default_retrieval_strategy')
    
    # Rest of search logic...
```

### Phase 5: Testing and Validation

#### Step 5.1: Update Test Files
For each test file using `--strategy`:
```python
# Replace patterns like:
- cmd = ["python", "cli.py", "ingest", "data.csv", "--strategy", "simple"]
+ cmd = ["python", "cli.py", "--config", "test_llamafarm.yaml", "ingest", "data.csv", "--database", "test_db", "--data-processing-strategy", "test_processing"]

- cmd = ["python", "cli.py", "search", "query", "--strategy", "simple"]  
+ cmd = ["python", "cli.py", "--config", "test_llamafarm.yaml", "search", "query", "--database", "test_db"]
```

#### Step 5.2: Integration Testing
```python
def test_new_cli_args():
    """Test new CLI argument structure."""
    # Test required arguments
    result = run_cli(["--config", "test_llamafarm.yaml", "ingest", "data.csv"])  # Should fail
    assert "database argument is required" in result.stderr
    
    # Test valid usage
    result = run_cli(["--config", "test_llamafarm.yaml", "ingest", "data.csv", "--database", "test_db", "--data-processing-strategy", "test_proc"])
    assert result.returncode == 0
    
    # Test search with optional retrieval strategy
    result = run_cli(["--config", "test_llamafarm.yaml", "search", "query", "--database", "test_db", "--retrieval-strategy", "custom"])
    assert result.returncode == 0
```

#### Step 5.3: Backward Compatibility Testing
```python
def test_legacy_arg_errors():
    """Test that legacy arguments show helpful errors."""
    result = run_cli(["ingest", "data.csv", "--strategy", "simple"])
    assert "strategy argument is no longer supported" in result.stderr
    assert "Use --config llamafarm.yaml --database" in result.stderr
```

---

## Global Configuration Integration Summary

### 🎯 Key Global Config Features

1. **Unified Configuration**: Single `llamafarm.yaml` or `llamafarm.toml` for entire project
2. **Schema Composition**: RAG schema embedded via `$ref: "../rag/schema.yaml"`
3. **Auto-detection**: CLI finds global config files automatically
4. **Validation**: Full schema validation using `/config/schema.yaml`
5. **Global Config Only**: No legacy RAG-specific configs supported

### 📋 Global Config Structure
```yaml
version: v1
name: my-project
namespace: my-namespace

# Project-wide settings
runtime:
  provider: ollama
  model: llama3.2

# Optional: Dataset definitions for tracking processed files
datasets: []

# RAG configuration (embedded schema)
rag:
  databases: [...]                     # Vector stores with embedding/retrieval strategies  
  data_processing_strategies: [...]    # File processing pipelines
```

### 🚀 User Experience Improvements

**Clean, Explicit Commands**:
```bash
# Direct database and strategy control
python cli.py --config llamafarm.yaml ingest docs/ --database main_database --data-processing-strategy text_processing
python cli.py --config llamafarm.yaml search "query" --database main_database --retrieval-strategy advanced_search
```

**Auto-detection**:
```bash
# No --config needed if llamafarm.yaml is in current directory
python cli.py ingest docs/ --database main_database --data-processing-strategy text_processing
python cli.py search "query" --database main_database
```

---

**Total Estimated Time**: 6-10 days (increased due to global config integration)  
**Risk Level**: HIGH (due to breaking changes)  
**Complexity**: HIGH (global config integration adds complexity)  
**Priority**: CRITICAL (user requirement + global config requirement)

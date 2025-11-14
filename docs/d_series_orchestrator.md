# D-Series Orchestrator

Master orchestrator implementing the MetaCore v4.4.1 workflow specification for the Revenant platform.

## Architecture

The D-Series Orchestrator executes workflow definitions composed of multiple node types:

- **call_agent**: Invoke other agents in the registry
- **evaluate**: Decision points using EvaluatorAgent
- **fuse**: Combine multiple outputs using FusionAgent  
- **store_memory**: Persist data using MemoryAgent
- **route**: Conditional branching
- **external_http**: External API calls (mock)

## Features

- **Async Execution**: Full asynchronous node execution with concurrency
- **Circuit Breaker**: Automatic failure protection for agent calls
- **Retry Logic**: Exponential backoff with configurable retries
- **Metrics Collection**: Per-node timing and status tracking
- **Trace Propagation**: End-to-end trace_id correlation
- **Timeout Management**: Per-node and global timeouts
- **Dependency Resolution**: Parallel execution of independent nodes

## Usage

### Basic Instantiation

```python
from agents.d_series.orchestrator import DSeriesOrchestrator

# Create orchestrator with default dependencies
orchestrator = DSeriesOrchestrator()

# Or with custom dependencies
orchestrator = DSeriesOrchestrator(
    registry=my_registry,
    storage=my_storage,
    logger=my_logger
)
# Agentform™ Schema

Core data models and YAML schemas for Agentform™ (Declarative AI agent framework).

## Installation

```bash
poetry install
```

## Usage

```python
from agentform_schema import SpecRoot, parse_yaml

spec = SpecRoot.model_validate(yaml_data)
```


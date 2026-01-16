# Agentform Schema

Core data models and YAML schemas for Agentform (Agent as code protocol).

## Installation

```bash
poetry install
```

## Usage

```python
from agentform_schema import SpecRoot, parse_yaml

spec = SpecRoot.model_validate(yaml_data)
```


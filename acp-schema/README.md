# ACP Schema

Core data models and YAML schemas for ACP (Agent as code protocol).

## Installation

```bash
poetry install
```

## Usage

```python
from acp_schema import SpecRoot, parse_yaml

spec = SpecRoot.model_validate(yaml_data)
```


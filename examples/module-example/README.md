# Module Example

This example demonstrates how to use ACP modules to create reusable configurations.

## Structure

```
module-example/
├── modules/
│   └── simple-acp-module/      # Reusable ACP module
│       ├── 00-project.acp      # Module metadata
│       ├── 01-variables.acp    # Module parameters (inputs)
│       ├── 02-providers.acp    # OpenAI provider configuration
│       ├── 03-policies.acp     # Standard and strict policies
│       ├── 04-models.acp       # Pre-configured models
│       ├── 05-agents.acp       # Pre-configured agents
│       └── 06-workflows.acp    # Ready-to-use workflows
├── 00-project.acp              # Main project metadata
├── 01-variables.acp            # Project variables
├── 02-modules.acp              # Module imports
├── input.yaml                  # Example input
└── README.md
```

## The Module

The `simple-acp-module` in `./modules/simple-acp-module/` provides everything you need:

- **Provider**: Pre-configured OpenAI provider
- **Models**: 
  - `default` - General purpose (configurable model ID)
  - `creative` - High temperature for creative tasks
  - `precise` - Low temperature for accurate responses
- **Policies**:
  - `standard` - Configurable cost and timeout limits
  - `strict` - Low cost limit and short timeout
- **Agents**:
  - `quick_assistant` - A concise, quick-response agent
- **Workflows**:
  - `quick_ask` - Simple Q&A workflow
  - `summarize` - Text summarization workflow

### Module Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `api_key` | string | Yes | - | OpenAI API key |
| `default_model` | string | No | "gpt-4o-mini" | Model ID for the default model |
| `max_cost_per_run` | number | No | 1.00 | Max cost in USD per run |
| `timeout_seconds` | number | No | 120 | Timeout for LLM calls |

## Using the Module

Import the module and pass required paraexpometers:

```acp
module "llm" {
  source = "./modules/simple-acp-module"
  
  api_key          = var.openai_api_key
  default_model    = "gpt-4o-mini"
  max_cost_per_run = 2.00
}
```

Then run the module's workflows directly:

```bash
# Run the quick_ask workflow from the module
acp run . --workflow module.llm.quick_ask --var openai_api_key=$OPENAI_API_KEY

# Run the summarize workflow
acp run . --workflow module.llm.summarize --var openai_api_key=$OPENAI_API_KEY
```

## Running

```bash
# Validate the configuration
acp validate .

# Run with input file
acp run . --workflow module.llm.quick_ask \
  --var openai_api_key=$OPENAI_API_KEY \
  --input input.yaml
```

## Benefits of Modules

1. **Reusability**: Share complete workflows across projects
2. **Encapsulation**: Hide implementation details behind a clean interface
3. **Zero Configuration**: Import a module and immediately use its workflows
4. **Consistency**: Ensure all projects use the same provider settings
5. **Maintainability**: Update module once, all consumers benefit

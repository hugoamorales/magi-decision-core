# MAGI Decision Core v5

## Overview
MAGI Decision Core v5 is a multi-agent deliberation system inspired by *Neon Genesis Evangelion*'s MAGI system. It coordinates multiple AI agents (Melchior, Balthasar, Casper, and optionally Gabriel) to deliberate on a user-provided problem statement, seeking consensus or selecting a winner through advanced voting. The system leverages APIs from OpenAI, Anthropic, and xAI to power its agents, providing a robust framework for decision-making.

### Key Features
- Dynamic agent configuration via `config.json`.
- Supports multiple AI models (OpenAI, Anthropic, xAI) for agent interactions.
- Implements retry logic for API calls with error recovery.
- Provides output in JSON, Markdown, or HTML formats.
- Includes a lightweight mode for faster execution.
- Validates `.env` file permissions for security.
- Conditional usage of the Gabriel agent with user-specified models.

## Installation

### Prerequisites
- Python 3.8 or higher
- Required Python packages (listed in `requirements.txt`)
- API keys for OpenAI, Anthropic, and xAI (stored in a `.env` file)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/hugoamorales/magi-decision-core.git
   cd magi-decision-core
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your `.env` file with API keys:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` to include your API keys:
   ```
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_anthropic_key
   XAI_API_KEY=your_xai_key
   ```
   Ensure the `.env` file has secure permissions:
   ```bash
   chmod 600 .env
   ```
4. Install the package:
   ```bash
   pip install .
   ```

## Usage

### Quick Start with Pipeline Examples

**NEW!** Try our pre-built pipeline examples with interactive UI visualization:

```bash
cd examples

# Run Decision Maker Pipeline (strategic decision-making)
./run_decision_maker_example.sh

# Run Article Creator Pipeline (multi-perspective content creation)
./run_article_creator_example.sh
```

Each pipeline generates an **interactive HTML UI** that displays:
- ✅ Side-by-side agent outputs with color-coded perspectives
- ✅ Stage-by-stage progress tracking
- ✅ Confidence meters and consensus indicators
- ✅ Expandable/collapsible sections for easy navigation

**See [examples/QUICKSTART.md](examples/QUICKSTART.md) for a 5-minute getting started guide!**

### Command-Line Usage

Run the script using the `magi` command (available after installation) or directly with `python magi_system_v5.py`. The script requires a `--problem` argument to specify the problem for deliberation.

### Command-Line Options
- `--problem TEXT`: The problem statement for agents to solve (required).
- `--max-iterations INT`: Maximum deliberation iterations (default: 2).
- `--config PATH`: Path to configuration file (default: `config.json`).
- `--output-dir PATH`: Directory for output files (default: current directory).
- `--output-format [json|md|html]`: Output format (default: `json`).
- `--dry-run`: Run without real API calls (for testing).
- `--lightweight`: Disable rich/pandas for faster execution.
- `--use-gabriel`: Include the Gabriel agent in deliberation.
- `--gabriel-model [openai|anthropic|xai]`: Model for Gabriel (default: `openai`).
- `--agent-melchior MODEL`: Override model for Melchior.
- `--agent-balthasar MODEL`: Override model for Balthasar.
- `--agent-casper MODEL`: Override model for Casper.

### Friendly Help Message
If you run the script without a `--problem` argument, with `--help`, or if an error occurs, you’ll see a friendly message describing the script’s purpose, available options, and usage examples. For example:
```plaintext
Welcome to MAGI Decision Core v5!

This script coordinates multiple AI agents (Melchior, Balthasar, Casper, and optionally Gabriel) to deliberate on a user-provided problem statement...

Options Available:
- --problem TEXT: The problem statement for agents to solve (required).
...

Examples of Potential Use:
- Analyze a business strategy:
  python magi_system_v5.py --problem "How can we increase market share in Europe?" --output-format html
...
```

### Examples
1. **Analyze a Business Strategy**:
   ```bash
   python magi_system_v5.py --problem "How can we increase market share in Europe?" --output-format html
   ```
2. **Optimize a Technical Process in Dry-Run Mode**:
   ```bash
   python magi_system_v5.py --problem "How to reduce server downtime?" --dry-run --lightweight
   ```
3. **Explore Ethical Dilemmas with Gabriel**:
   ```bash
   python magi_system_v5.py --problem "What are the ethical implications of AI in healthcare?" --use-gabriel --gabriel-model anthropic
   ```

## Output
The script generates output files in the specified format (`json`, `md`, or `html`) containing:
- The problem statement.
- Iteration details and agent proposals.
- Consensus status and final solution.
- Voting results (if consensus is not reached).

Output files are saved with timestamps (e.g., `magi_output_20250604_155200.json`). The script now ensures proper file path handling, fixing previous issues with undefined variables.

## Pipeline System

MAGI Decision Core now includes a powerful pipeline system for orchestrating multi-stage deliberations with interactive visualization.

### Available Pipelines

1. **Decision Maker Pipeline** (`examples/pipelines/decision_maker_pipeline.yml`)
   - Strategic decision-making and problem-solving
   - Stages: Gather proposals → Consensus check → Cross-justification → Voting
   - Perfect for: Business decisions, technical choices, resource allocation

2. **Article Creator Pipeline** (`examples/pipelines/article_creator_pipeline.yml`)
   - Multi-perspective article generation
   - Stages: Topic analysis → Content strategy → Article drafting → Quality review → Final synthesis
   - Perfect for: Blog posts, documentation, marketing content

### Creating Custom Pipelines

1. Copy a pipeline template from `examples/pipelines/`
2. Customize stages, agents, and prompts in the YAML file
3. Run with: `python examples/pipeline_runner.py --pipeline your_pipeline.yml`

See [examples/README.md](examples/README.md) for detailed documentation.

### Interactive UI Features

The pipeline system generates rich HTML visualizations:
- **Side-by-side agent outputs**: Compare perspectives in real-time
- **Color-coded agents**: Blue (Analytical), Red (Human-centric), Green (Creative)
- **Stage progress**: Visual tracking of pipeline execution
- **Confidence meters**: See agent confidence levels
- **Consensus indicators**: Understand when agents agree/disagree

## Development Notes
- The script includes enhanced internal documentation with detailed docstrings and comments for better maintainability.
- The Gabriel agent is now optional and configurable, allowing users to specify its inclusion and model via command-line arguments.
- Error handling has been improved to provide user-friendly feedback.
- **NEW**: Pipeline system for orchestrating complex multi-stage deliberations with interactive UI.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## License
This project is licensed under the MIT License.

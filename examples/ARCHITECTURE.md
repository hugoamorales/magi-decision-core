# MAGI Pipeline System Architecture

## Overview

The MAGI Pipeline System orchestrates multi-agent deliberations through configurable stages, with each stage leveraging the unique perspectives of three AI agents (Melchior, Balthasar, and Casper) to produce comprehensive, well-rounded outputs.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Pipeline Runner                          │
│  (pipeline_runner.py)                                       │
│                                                             │
│  ┌──────────────┐                                          │
│  │ Load Config  │──► YAML Pipeline Definition              │
│  └──────────────┘                                          │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                          │
│  │ Initialize   │──► Connect to APIs (OpenAI, Anthropic,   │
│  │   Agents     │    xAI)                                  │
│  └──────────────┘                                          │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                          │
│  │  Execute     │──► Run Pipeline Stages Sequentially      │
│  │  Pipeline    │                                          │
│  └──────────────┘                                          │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                          │
│  │  Generate    │──► Create JSON, Markdown, HTML outputs   │
│  │  Outputs     │                                          │
│  └──────────────┘                                          │
└─────────────────────────────────────────────────────────────┘
```

## Agent Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Three Agent Triad                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐  │
│  │   Melchior     │  │   Balthasar    │  │    Casper    │  │
│  │   (Blue)       │  │     (Red)      │  │   (Green)    │  │
│  ├────────────────┤  ├────────────────┤  ├──────────────┤  │
│  │ GPT-4o         │  │ Claude 3.5     │  │   Grok-3     │  │
│  │                │  │  Sonnet        │  │              │  │
│  ├────────────────┤  ├────────────────┤  ├──────────────┤  │
│  │ Analytical     │  │ Human-Centric  │  │  Creative    │  │
│  │ Data-Driven    │  │ Safety-Focused │  │  Innovative  │  │
│  │ Optimization   │  │ Ethical        │  │  Value-Based │  │
│  └────────────────┘  └────────────────┘  └──────────────┘  │
│         │                    │                   │          │
│         └────────────────────┴───────────────────┘          │
│                              │                              │
│                    Consensus Mechanism                      │
│                              │                              │
│              ┌───────────────┴───────────────┐              │
│              │                               │              │
│      Semantic Similarity            Confidence Scoring      │
│       (0.8 weight)                    (0.2 weight)          │
│              │                               │              │
│              └───────────────┬───────────────┘              │
│                              │                              │
│                    Consensus Score ≥ 0.7?                   │
│                              │                              │
│                  ┌───────────┴───────────┐                  │
│                  │                       │                  │
│                 Yes                     No                  │
│                  │                       │                  │
│          Final Solution      Cross-Justification Round      │
│                                          │                  │
│                                   Advanced Voting           │
└──────────────────────────────────────────────────────────────┘
```

## Pipeline Workflows

### Decision Maker Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│              DECISION MAKER PIPELINE                        │
└─────────────────────────────────────────────────────────────┘

Input: Problem Statement
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Gather Proposals                                   │
│ ─────────────────────────                                   │
│                                                             │
│ Melchior, Balthasar, Casper (Parallel)                     │
│ Each provides:                                              │
│  • Problem Analysis                                         │
│  • Proposed Solution (Action Plan)                          │
│  • Expected Outcomes                                        │
│  • Risks and Mitigation                                     │
│  • Confidence Level                                         │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Consensus Check                                    │
│ ────────────────────────                                    │
│                                                             │
│ Calculate consensus score from:                             │
│  • Semantic similarity between proposals (80%)              │
│  • Average confidence levels (20%)                          │
│                                                             │
│ Threshold: 0.7                                              │
└─────────────────────────────────────────────────────────────┘
    │
    ├──── Consensus Reached ──────────────────────┐
    │                                              │
    └──── No Consensus                             ▼
         │                                    Final Solution
         ▼                                         │
┌─────────────────────────────────────────────────│───────────┐
│ Stage 3: Cross-Justification                    │           │
│ ────────────────────────────                    │           │
│                                                 │           │
│ Each agent reviews others' proposals            │           │
│  • Identifies strengths to incorporate          │           │
│  • Addresses gaps and weaknesses                │           │
│  • Refines their own proposal                   │           │
│  • Updates confidence level                     │           │
│                                                 │           │
│ Max iterations: 2                               │           │
└─────────────────────────────────────────────────│───────────┘
    │                                              │
    ▼                                              │
┌─────────────────────────────────────────────────│───────────┐
│ Stage 4: Voting (if still no consensus)         │           │
│ ────────────────────────────────────            │           │
│                                                 │           │
│ Each agent votes for best proposal              │           │
│  • Evaluates feasibility                        │           │
│  • Assesses comprehensiveness                   │           │
│  • Considers risk management                    │           │
│  • Measures expected impact                     │           │
│                                                 │           │
│ Winner selected by majority vote                │           │
└─────────────────────────────────────────────────│───────────┘
    │                                              │
    └──────────────────────────────────────────────┘
                        │
                        ▼
               Output Generation
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
     JSON.json      output.md      output.html
  (Raw Data)    (Readable Text)  (Interactive UI)
```

### Article Creator Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│              ARTICLE CREATOR PIPELINE                       │
└─────────────────────────────────────────────────────────────┘

Input: Idea, Target Audience, Tones, Length
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Topic Analysis                                     │
│ ───────────────────────                                     │
│                                                             │
│ Melchior, Balthasar, Casper (Parallel)                     │
│ Each provides:                                              │
│  • Topic Analysis (key themes, angles)                      │
│  • Audience Considerations                                  │
│  • Tone Recommendations                                     │
│  • Unique Angle from their perspective                      │
│  • Key Points to Cover                                      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Content Strategy                                   │
│ ─────────────────────────                                   │
│                                                             │
│ Melchior, Balthasar, Casper (Parallel)                     │
│ Each develops:                                              │
│  • Article Structure (outline with sections)                │
│  • Opening Hook                                             │
│  • Key Arguments/Points                                     │
│  • Supporting Elements (examples, data, stories)            │
│  • Conclusion Strategy                                      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Article Drafting                                   │
│ ──────────────────────────                                  │
│                                                             │
│ Melchior, Balthasar, Casper (Parallel)                     │
│ Each writes:                                                │
│  • Complete article draft                                   │
│  • Following agreed structure                               │
│  • Incorporating selected tones                             │
│  • Targeting specified audience                             │
│  • Meeting length requirements                              │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 4: Quality Review                                     │
│ ───────────────────────                                     │
│                                                             │
│ Melchior, Balthasar, Casper (Parallel)                     │
│ Each evaluates all drafts on:                               │
│  • Tone Consistency                                         │
│  • Audience Fit                                             │
│  • Content Quality                                          │
│  • Readability                                              │
│  • Recommendation for final version                         │
│                                                             │
│ Consensus threshold: 0.75                                   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 5: Final Synthesis                                    │
│ ────────────────────────                                    │
│                                                             │
│ Melchior synthesizes:                                       │
│  • Best elements from each draft                            │
│  • Quality review feedback                                  │
│  • Final polished article                                   │
│  • Title and subtitle options                               │
│  • Synthesis notes                                          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
               Output Generation
                    │
    ┌───────────────┼───────────────┐
    │               │               │
    ▼               ▼               ▼
 JSON.json      output.md      output.html
(All Drafts)  (Side-by-Side)  (Interactive UI)
```

## UI Generation System

```
┌─────────────────────────────────────────────────────────────┐
│                   UI Generator                              │
│                (ui_generator.py)                            │
└─────────────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
   HTML Header   Stage Sections   Agent Proposals
        │               │               │
        │       ┌───────┴────────┐      │
        │       ▼                ▼      │
        │  Progress Bar    Consensus    │
        │                   Info        │
        │                               │
        └───────────────┬───────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  Interactive Features │
            ├───────────────────────┤
            │ • Collapsible stages  │
            │ • Color-coded agents  │
            │ • Confidence meters   │
            │ • Side-by-side view   │
            │ • Smooth animations   │
            └───────────────────────┘
                        │
                        ▼
              Complete HTML Document
                        │
                        ▼
                  output.html
              (Open in Browser)
```

## Data Flow

```
User Input
    │
    ▼
Pipeline Config (YAML)
    │
    ├──► Pipeline Definition
    │    • Name, Description
    │    • Pipeline Type
    │    • Input Schema
    │    • Stage Definitions
    │    • UI Configuration
    │
    ▼
Pipeline Runner
    │
    ├──► Initialize Agents
    │    • Load config.json
    │    • Connect to APIs
    │    • Set up retry logic
    │
    ├──► Execute Stages
    │    │
    │    ├──► Stage 1
    │    │    • Format prompts
    │    │    • Call agents (parallel)
    │    │    • Parse responses
    │    │    • Store outputs
    │    │
    │    ├──► Stage 2
    │    │    • Check consensus
    │    │    • Calculate scores
    │    │    • Store results
    │    │
    │    └──► Stage N...
    │
    └──► Generate Outputs
         │
         ├──► JSON (raw data)
         ├──► Markdown (readable)
         └──► HTML (interactive UI)
              │
              └──► UI Generator
                   • Parse results
                   • Apply styling
                   • Add interactivity
                   • Generate HTML
```

## Configuration Files

### Pipeline YAML Structure

```yaml
name: "Pipeline Name"
description: "What it does"
pipeline_type: "identifier"

inputs:
  - name: "param_name"
    type: "text|select|multi_select"
    required: true

pipeline_stages:
  - stage: "stage_name"
    description: "What this stage does"
    agents: ["Agent1", "Agent2"]
    prompt_template: "Prompt with {variables}"
    timeout: 60
    consensus_threshold: 0.7

ui_display:
  layout: "side_by_side"
  show_agent_colors: true
  agent_colors:
    Melchior: "#3498db"
    Balthasar: "#e74c3c"
    Casper: "#2ecc71"
```

### Agent Config (config.json)

```json
{
  "agents": {
    "Melchior": {
      "model": "gpt-4o",
      "api": "openai",
      "role": "Analytical agent..."
    },
    "Balthasar": {
      "model": "claude-3-5-sonnet-20241022",
      "api": "anthropic",
      "role": "Human-centric agent..."
    },
    "Casper": {
      "model": "grok-3",
      "api": "xai",
      "role": "Creative agent..."
    }
  },
  "api_settings": {
    "retry_attempts": 3,
    "retry_min_wait": 1,
    "retry_max_wait": 5,
    "consensus_threshold": 0.7
  }
}
```

## Key Design Principles

1. **Modularity**: Each stage is independent and composable
2. **Parallelization**: Agent calls within stages run concurrently
3. **Flexibility**: YAML-based configuration for easy customization
4. **Transparency**: Full visibility into each agent's reasoning
5. **Visualization**: Rich UI for comparing perspectives
6. **Extensibility**: Easy to add new pipelines and stages

## Performance Considerations

- **Async Operations**: All API calls are asynchronous
- **Parallel Execution**: Agents run simultaneously within stages
- **Caching**: Results cached at each stage
- **Retry Logic**: Automatic retry with exponential backoff
- **Timeout Management**: Configurable timeouts per stage

## Security

- **API Key Protection**: Keys stored in .env with restricted permissions
- **Input Validation**: All user inputs validated
- **Error Handling**: Comprehensive error handling throughout
- **Sandboxing**: Agent responses parsed and sanitized

---

For implementation details, see:
- [README.md](README.md) - User guide
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [pipeline_runner.py](pipeline_runner.py) - Main implementation
- [ui_generator.py](ui_generator.py) - UI generation logic

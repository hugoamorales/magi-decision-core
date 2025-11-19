# MAGI Pipeline Examples

This directory contains example pipelines demonstrating the MAGI Decision Core system's capabilities for multi-agent deliberation and content creation.

## 📋 Available Pipelines

### 1. Decision Maker Pipeline
**Purpose:** Strategic decision-making and problem-solving through multi-agent deliberation

**Use Cases:**
- Business strategy decisions
- Technical architecture choices
- Resource allocation problems
- Risk assessment and mitigation planning

**How it works:**
1. **Gather Proposals:** Each agent (Melchior, Balthasar, Casper) analyzes the problem and proposes an action plan
2. **Consensus Check:** Evaluates if agents reached consensus on the approach
3. **Cross-Justification:** Agents refine their proposals based on others' perspectives
4. **Voting:** If no consensus, agents vote for the best solution

**Output:** Comprehensive action plan with analysis, steps, risks, and confidence metrics

---

### 2. Article Creator Pipeline
**Purpose:** Multi-perspective article generation with tone and audience customization

**Use Cases:**
- Blog posts and articles
- Technical documentation
- Marketing content
- Educational materials

**How it works:**
1. **Topic Analysis:** Agents analyze the topic from different perspectives
2. **Content Strategy:** Develop structure and approach
3. **Article Drafting:** Each agent writes a complete draft
4. **Quality Review:** Cross-review and consensus on quality
5. **Final Synthesis:** Combine best elements into final article

**Output:** Polished article with multiple perspectives and comprehensive quality review

---

## 🚀 Quick Start

### Prerequisites

1. Install dependencies:
```bash
pip install -r ../requirements.txt
```

2. Set up your API keys in `.env`:
```bash
cp ../.env_template ../.env
# Edit .env and add your API keys:
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# XAI_API_KEY=xai-...
```

### Running the Decision Maker Pipeline

```bash
python pipeline_runner.py \
  --pipeline pipelines/decision_maker_pipeline.yml \
  --problem "How should we expand our SaaS product into the enterprise market?" \
  --output-dir ./results/decision_maker
```

### Running the Article Creator Pipeline

```bash
python pipeline_runner.py \
  --pipeline pipelines/article_creator_pipeline.yml \
  --idea "The Future of AI in Healthcare" \
  --audience "Healthcare professionals and administrators" \
  --tones "Professional, Educational, Authoritative" \
  --length "Long (1500-2500 words)" \
  --output-dir ./results/article_creator
```

---

## 📊 UI Visualization

Each pipeline run generates an interactive HTML visualization that displays:

### Features:
- ✅ **Side-by-Side Agent Outputs:** Compare proposals from Melchior, Balthasar, and Casper
- ✅ **Stage Progress Tracking:** Visual representation of pipeline execution
- ✅ **Confidence Meters:** See each agent's confidence in their proposals
- ✅ **Consensus Indicators:** Understand when agents agree or disagree
- ✅ **Collapsible Stages:** Expand/collapse sections for easy navigation
- ✅ **Color-Coded Agents:**
  - 🔵 **Melchior** (Blue): Analytical, data-driven perspective
  - 🔴 **Balthasar** (Red): Human-centric, accessibility perspective
  - 🟢 **Casper** (Green): Creative, innovative perspective

### Output Files:
After running a pipeline, you'll get three files:
- `{pipeline_type}_{timestamp}.json` - Raw data in JSON format
- `{pipeline_type}_{timestamp}.md` - Human-readable Markdown format
- `{pipeline_type}_{timestamp}.html` - **Interactive UI** (open in browser)

---

## 🛠️ Customization

### Creating Custom Pipelines

1. **Copy a template:**
```bash
cp pipelines/decision_maker_pipeline.yml pipelines/my_custom_pipeline.yml
```

2. **Edit the configuration:**
   - Change `name` and `description`
   - Modify `inputs` to match your needs
   - Customize `pipeline_stages` and prompts
   - Adjust `ui_display` settings

3. **Run your custom pipeline:**
```bash
python pipeline_runner.py \
  --pipeline pipelines/my_custom_pipeline.yml \
  --your-custom-input "value"
```

### Pipeline Configuration Reference

```yaml
name: "Pipeline Name"
description: "What this pipeline does"
pipeline_type: "unique_identifier"

inputs:
  - name: "input_name"
    type: "text|select|multi_select"
    required: true
    description: "Input description"

pipeline_stages:
  - stage: "stage_name"
    description: "What this stage does"
    agents: ["Melchior", "Balthasar", "Casper"]
    timeout: 60
    prompt_template: |
      Your prompt here with {variables}

ui_display:
  layout: "side_by_side"
  show_agent_colors: true
  show_stage_progress: true
```

---

## 🎨 Agent Personalities

Understanding how each agent approaches problems:

### Melchior (Analytical)
- **Model:** GPT-4o
- **Focus:** Data-driven solutions, technical optimization, efficiency
- **Strengths:** Quantitative analysis, systematic approaches, scalability
- **Best for:** Technical decisions, data analysis, optimization problems

### Balthasar (Human-Centric)
- **Model:** Claude 3.5 Sonnet
- **Focus:** Human welfare, emotional stability, safety, accessibility
- **Strengths:** User experience, ethical considerations, risk mitigation
- **Best for:** User-facing decisions, safety-critical systems, ethical dilemmas

### Casper (Creative)
- **Model:** Grok-3
- **Focus:** Innovation, creative solutions, ethical values
- **Strengths:** Out-of-the-box thinking, novel approaches, value alignment
- **Best for:** Innovation challenges, creative content, strategic vision

---

## 📖 Example Use Cases

### Decision Maker Examples:

1. **Product Strategy:**
```bash
--problem "Should we build our own infrastructure or use cloud providers?"
```

2. **Team Organization:**
```bash
--problem "How should we restructure our engineering team to improve velocity?"
```

3. **Technology Choice:**
```bash
--problem "Which database technology should we use for our real-time analytics platform?"
```

### Article Creator Examples:

1. **Technical Blog Post:**
```bash
--idea "Introduction to Microservices Architecture" \
--audience "Software engineers and architects" \
--tones "Technical, Educational, Professional"
```

2. **Marketing Content:**
```bash
--idea "Why Your Business Needs AI Automation" \
--audience "Small business owners" \
--tones "Conversational, Persuasive, Professional"
```

3. **Educational Material:**
```bash
--idea "Understanding Climate Change Impact" \
--audience "High school students" \
--tones "Educational, Conversational, Inspirational"
```

---

## 🔧 Advanced Options

### Using Different Agent Models

Edit `../config.json` to use different models:

```json
{
  "agents": {
    "Melchior": {
      "model": "gpt-4o",
      "api": "openai"
    },
    "Balthasar": {
      "model": "claude-3-5-sonnet-20241022",
      "api": "anthropic"
    },
    "Casper": {
      "model": "grok-3",
      "api": "xai"
    }
  }
}
```

### Adjusting Consensus Threshold

In your pipeline YAML:

```yaml
pipeline_stages:
  - stage: "consensus_check"
    consensus_threshold: 0.75  # Higher = stricter consensus
```

### Custom Output Formats

Specify output formats in pipeline YAML:

```yaml
output_format: ["json", "md", "html"]
# Available: json, md, html
```

---

## 📂 Directory Structure

```
examples/
├── README.md                          # This file
├── pipeline_runner.py                 # Main runner script
├── ui_generator.py                    # UI generation module
├── pipelines/                         # Pipeline configurations
│   ├── decision_maker_pipeline.yml
│   └── article_creator_pipeline.yml
├── decision_maker/                    # Decision maker examples
│   └── example_runs/
├── article_creator/                   # Article creator examples
│   └── example_runs/
└── ui_templates/                      # Custom UI templates
```

---

## 🐛 Troubleshooting

### API Key Errors
```
Error: OPENAI_API_KEY not found
```
**Solution:** Check that your `.env` file exists and contains valid API keys.

### Import Errors
```
ModuleNotFoundError: No module named 'yaml'
```
**Solution:** Install dependencies: `pip install -r ../requirements.txt`

### No Output Generated
**Solution:** Check the `--output-dir` path exists and you have write permissions.

---

## 📝 Best Practices

1. **Clear Problem Statements:** Be specific and detailed in your inputs
2. **Appropriate Audience:** Clearly define who you're writing for
3. **Tone Selection:** Choose 2-3 complementary tones for best results
4. **Review Outputs:** Use the interactive UI to compare agent perspectives
5. **Iterate:** Run pipelines multiple times with refined inputs

---

## 🤝 Contributing

Want to add a new pipeline? Follow these steps:

1. Create a new YAML file in `pipelines/`
2. Define inputs, stages, and prompts
3. Test with `pipeline_runner.py`
4. Share your pipeline!

---

## 📧 Support

For issues or questions:
- Check the [main README](../README.md)
- Review pipeline YAML configurations
- Examine example outputs in `results/`

---

## 🎯 Next Steps

1. Try both example pipelines
2. Open the generated HTML files in your browser
3. Compare agent outputs side-by-side
4. Experiment with different inputs and tones
5. Create your own custom pipeline

**Happy deliberating! 🧠**

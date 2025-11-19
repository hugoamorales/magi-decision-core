#!/usr/bin/env python3
"""
MAGI Pipeline Runner
Executes decision-making and article creation pipelines with live UI visualization.
"""

import asyncio
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys

# Add parent directory to path to import magi_system_v5
sys.path.insert(0, str(Path(__file__).parent.parent))

from magi_system_v5 import (
    get_agent_config,
    initialize_agents,
    call_openai,
    call_anthropic,
    call_xai,
    parse_response,
    check_consensus,
    cross_justification_round,
    advanced_vote,
    save_output
)


class PipelineRunner:
    """Runs MAGI pipelines with stage tracking and UI generation."""

    def __init__(self, pipeline_config_path: str, config_path: str = "config.json"):
        self.pipeline_config = self._load_pipeline_config(pipeline_config_path)
        self.config_path = config_path
        self.agents = None
        self.stage_outputs = {}
        self.current_stage = None
        self.start_time = None

    def _load_pipeline_config(self, path: str) -> Dict:
        """Load pipeline configuration from YAML file."""
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    async def initialize(self):
        """Initialize agents and prepare for execution."""
        agent_config = get_agent_config(self.config_path)
        self.agents = initialize_agents(agent_config)
        print(f"✓ Initialized {len(self.agents)} agents")

    async def call_agent(self, agent_name: str, prompt: str) -> str:
        """Call a specific agent with retry logic."""
        agent = self.agents[agent_name]

        try:
            if agent["api"] == "openai":
                response = await call_openai(prompt, agent)
            elif agent["api"] == "anthropic":
                response = await call_anthropic(prompt, agent)
            elif agent["api"] == "xai":
                response = await call_xai(prompt, agent)
            else:
                raise ValueError(f"Unknown API type: {agent['api']}")

            return response
        except Exception as e:
            print(f"Error calling {agent_name}: {e}")
            raise

    def format_prompt(self, template: str, variables: Dict[str, Any]) -> str:
        """Format prompt template with variables."""
        return template.format(**variables)

    async def run_stage(self, stage_config: Dict, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single pipeline stage."""
        stage_name = stage_config["stage"]
        self.current_stage = stage_name

        print(f"\n{'='*60}")
        print(f"Stage: {stage_name}")
        print(f"Description: {stage_config['description']}")
        print(f"{'='*60}\n")

        stage_output = {
            "stage": stage_name,
            "description": stage_config["description"],
            "timestamp": datetime.now().isoformat(),
            "proposals": []
        }

        # Get agents for this stage
        stage_agents = stage_config.get("agents", list(self.agents.keys())[:3])

        # Check if this is a consensus check stage
        if stage_name == "consensus_check":
            if "previous_proposals" in context:
                consensus, details = check_consensus(context["previous_proposals"])
                stage_output["consensus"] = consensus
                stage_output["details"] = details
                context["consensus_reached"] = consensus
                print(f"Consensus: {'✓ Reached' if consensus else '✗ Not reached'}")
                print(f"Score: {details.get('consensus_score', 0):.3f}")
            return stage_output

        # Get prompt template
        prompt_template = stage_config.get("prompt_template", "")
        if not prompt_template:
            print(f"Warning: No prompt template for stage {stage_name}")
            return stage_output

        # Prepare context variables
        prompt_vars = context.copy()

        # Format prompts for each agent
        tasks = []
        for agent_name in stage_agents:
            # Customize prompt variables per agent if needed
            agent_vars = prompt_vars.copy()

            # Format other proposals for cross-justification
            if "previous_proposals" in context:
                other_proposals = "\n\n".join([
                    f"Agent {p['agent']}:\n{p.get('solution', p.get('response', ''))}"
                    for p in context["previous_proposals"]
                    if p["agent"] != agent_name
                ])
                agent_vars["other_proposals"] = other_proposals
                agent_vars["all_proposals"] = "\n\n".join([
                    f"Agent {p['agent']}:\n{p.get('solution', p.get('response', ''))}"
                    for p in context["previous_proposals"]
                ])

            prompt = self.format_prompt(prompt_template, agent_vars)
            tasks.append(self.call_agent_with_parse(agent_name, prompt))

        # Execute all agent calls in parallel
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Process responses
        for agent_name, response in zip(stage_agents, responses):
            if isinstance(response, Exception):
                print(f"Error from {agent_name}: {response}")
                continue

            stage_output["proposals"].append({
                "agent": agent_name,
                "response": response.get("raw_response", ""),
                "solution": response.get("solution", ""),
                "justification": response.get("justification", ""),
                "confidence": response.get("confidence", 0.0)
            })

            print(f"\n{agent_name} ({response.get('confidence', 0):.2f} confidence):")
            print(f"{response.get('solution', response.get('raw_response', ''))[:200]}...")

        return stage_output

    async def call_agent_with_parse(self, agent_name: str, prompt: str) -> Dict[str, Any]:
        """Call agent and parse response."""
        raw_response = await self.call_agent(agent_name, prompt)
        parsed = parse_response(raw_response, agent_name)
        parsed["raw_response"] = raw_response
        return parsed

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the complete pipeline."""
        self.start_time = datetime.now()

        print(f"\n{'#'*60}")
        print(f"# {self.pipeline_config['name']}")
        print(f"# {self.pipeline_config['description']}")
        print(f"{'#'*60}\n")

        # Initialize context with inputs
        context = inputs.copy()
        context["pipeline_name"] = self.pipeline_config["name"]

        results = {
            "pipeline": self.pipeline_config["name"],
            "pipeline_type": self.pipeline_config.get("pipeline_type", "unknown"),
            "inputs": inputs,
            "stages": [],
            "start_time": self.start_time.isoformat(),
            "end_time": None
        }

        # Execute each stage
        for stage_config in self.pipeline_config["pipeline_stages"]:
            # Check trigger conditions
            if "trigger_condition" in stage_config:
                if stage_config["trigger_condition"] == "no_consensus":
                    if context.get("consensus_reached", False):
                        print(f"Skipping {stage_config['stage']} (consensus reached)")
                        continue

            # Run the stage
            stage_output = await self.run_stage(stage_config, context)
            results["stages"].append(stage_output)

            # Update context with stage output
            if stage_output["proposals"]:
                context["previous_proposals"] = stage_output["proposals"]

                # Store stage-specific outputs
                stage_name = stage_config["stage"]
                if stage_name == "topic_analysis":
                    context["topic_analysis"] = stage_output["proposals"]
                elif stage_name == "content_strategy":
                    context["content_strategy"] = stage_output["proposals"]
                elif stage_name == "article_drafting":
                    context["all_drafts"] = stage_output["proposals"]
                elif stage_name == "quality_review":
                    context["quality_reviews"] = stage_output["proposals"]

        results["end_time"] = datetime.now().isoformat()
        results["duration_seconds"] = (datetime.now() - self.start_time).total_seconds()

        return results

    def generate_ui(self, results: Dict[str, Any], output_path: str):
        """Generate interactive UI for visualizing results."""
        from ui_generator import generate_pipeline_ui

        ui_config = self.pipeline_config.get("ui_display", {})
        html_content = generate_pipeline_ui(results, ui_config)

        with open(output_path, 'w') as f:
            f.write(html_content)

        print(f"\n✓ UI generated: {output_path}")

    def save_results(self, results: Dict[str, Any], output_dir: str):
        """Save results in multiple formats."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pipeline_type = results.get("pipeline_type", "pipeline")

        # Save JSON
        json_file = output_path / f"{pipeline_type}_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Saved JSON: {json_file}")

        # Save Markdown
        md_file = output_path / f"{pipeline_type}_{timestamp}.md"
        self._save_markdown(results, md_file)
        print(f"✓ Saved Markdown: {md_file}")

        # Generate and save UI
        ui_file = output_path / f"{pipeline_type}_{timestamp}.html"
        self.generate_ui(results, str(ui_file))

        return json_file, md_file, ui_file

    def _save_markdown(self, results: Dict[str, Any], output_path: Path):
        """Save results as markdown."""
        with open(output_path, 'w') as f:
            f.write(f"# {results['pipeline']}\n\n")
            f.write(f"**Started:** {results['start_time']}\n\n")
            f.write(f"**Duration:** {results['duration_seconds']:.2f}s\n\n")

            f.write("## Inputs\n\n")
            for key, value in results['inputs'].items():
                f.write(f"- **{key}:** {value}\n")
            f.write("\n")

            f.write("## Pipeline Execution\n\n")
            for stage in results['stages']:
                f.write(f"### {stage['stage']}\n\n")
                f.write(f"*{stage['description']}*\n\n")

                if "consensus" in stage:
                    f.write(f"**Consensus:** {'✓ Reached' if stage['consensus'] else '✗ Not reached'}\n\n")
                    if "details" in stage:
                        f.write(f"**Score:** {stage['details'].get('consensus_score', 0):.3f}\n\n")

                for proposal in stage.get('proposals', []):
                    f.write(f"#### {proposal['agent']}")
                    if 'confidence' in proposal:
                        f.write(f" (Confidence: {proposal['confidence']:.2f})")
                    f.write("\n\n")

                    content = proposal.get('solution') or proposal.get('response', '')
                    f.write(f"{content}\n\n")

                    if proposal.get('justification'):
                        f.write(f"**Justification:** {proposal['justification']}\n\n")

                f.write("---\n\n")


async def main():
    parser = argparse.ArgumentParser(description="Run MAGI pipelines")
    parser.add_argument("--pipeline", required=True, help="Path to pipeline YAML config")
    parser.add_argument("--config", default="config.json", help="Path to agent config")
    parser.add_argument("--output-dir", default="./pipeline_results", help="Output directory")

    # Pipeline-specific inputs
    parser.add_argument("--problem", help="Problem statement (for decision_maker)")
    parser.add_argument("--idea", help="Article idea (for article_creator)")
    parser.add_argument("--audience", help="Target audience (for article_creator)")
    parser.add_argument("--tones", help="Comma-separated tones (for article_creator)")
    parser.add_argument("--length", default="Medium (800-1500 words)", help="Article length")

    args = parser.parse_args()

    # Prepare inputs based on pipeline type
    pipeline_path = Path(args.pipeline)
    with open(pipeline_path, 'r') as f:
        pipeline_config = yaml.safe_load(f)

    inputs = {}
    pipeline_type = pipeline_config.get("pipeline_type", "unknown")

    if pipeline_type == "decision_maker":
        if not args.problem:
            print("Error: --problem is required for decision_maker pipeline")
            return
        inputs["problem"] = args.problem

    elif pipeline_type == "article_creator":
        if not args.idea:
            print("Error: --idea is required for article_creator pipeline")
            return
        inputs["idea"] = args.idea
        inputs["target_audience"] = args.audience or "general audience"
        inputs["tones"] = args.tones or "Professional, Educational"
        inputs["length"] = args.length

    # Run pipeline
    runner = PipelineRunner(args.pipeline, args.config)
    await runner.initialize()

    results = await runner.execute(inputs)

    # Save results
    json_file, md_file, ui_file = runner.save_results(results, args.output_dir)

    print(f"\n{'='*60}")
    print("Pipeline execution complete!")
    print(f"Open {ui_file} in your browser to view the interactive results.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())

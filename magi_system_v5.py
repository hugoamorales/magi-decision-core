"""
File: magi_system_v5.py
Description: Main script for MAGI Decision Core v5, a multi-agent deliberation system inspired by Neon Genesis Evangelion's MAGI.
Purpose: Coordinates multiple AI agents to deliberate on a given problem, seeking consensus or selecting a winner through voting.
Key Features:
- Dynamic agent configuration via config.json.
- Supports OpenAI, Anthropic, and xAI APIs for agent interactions.
- Implements retry logic for API calls with error recovery.
- Provides detailed output in JSON, Markdown, or HTML formats.
- Includes lightweight mode to disable rich/pandas for performance.
- Validates .env file permissions for security.
- Allows conditional usage of additional agents like Gabriel.
Modifications:
- Updated to display a friendly message with purpose, options, and examples when no arguments are provided or on --help.
- Added error handling to show the same friendly message on execution errors.
- Added conditional logic for Gabriel agent usage with model selection.
- Fixed output issue with base_filename variable scope.
- Enhanced internal documentation with detailed docstrings and comments.
- Moved AGENTS and API_SETTINGS initialization earlier to fix NameError.
- Adjusted error handling to catch early script errors.
- Added subfolder structure for results: "analysis_results/<problem_folder>".
- Fixed HTML output generation by removing incorrect float conversion of consensus_html string.
- Moved confidence_log.json and magi_system.log to the analysis_results/<problem_folder> subfolder.
- Added notification when no consensus is reached, informing users that potential solutions are stored in the results file.
- Fixed f-string backslash issue in cross_justification_round for Python 3.10 compatibility by moving join logic outside f-string.
"""
import argparse
import asyncio
import json
import os
import re
import sys
import stat  # For file permission checking
from typing import Dict, List, Tuple
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import aiohttp
from datetime import datetime
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import aiohttp.client_exceptions
from sentence_transformers import SentenceTransformer, util
import markdown
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.status import Status
from rich import print as rprint

# Custom exceptions for improved error handling
class OpenAIError(Exception):
    """Exception raised for OpenAI API errors."""
    pass

class AnthropicError(Exception):
    """Exception raised for Anthropic API errors."""
    pass

class XAIError(Exception):
    """Exception raised for xAI API errors."""
    pass

class MissingAPIKeyError(Exception):
    """Exception raised when required API keys are missing."""
    pass

class InsecureEnvFileError(Exception):
    """Exception raised when .env file has insecure permissions."""
    pass

# Define the friendly message to display for help, missing arguments, or errors
HELP_MESSAGE = """
[bold blue]Welcome to MAGI Decision Core v5![/bold blue]

This script coordinates multiple AI agents (Melchior, Balthasar, Casper, and optionally Gabriel) to deliberate on a user-provided problem statement. Each agent proposes solutions based on their unique roles, and the system seeks consensus or selects a winner through advanced voting. The process is inspired by *Neon Genesis Evangelion*'s MAGI system.

[bold]Options Available:[/bold]
- --problem TEXT: The problem statement for agents to solve (required).
- --max-iterations INT: Maximum deliberation iterations (default: 2).
- --config PATH: Path to configuration file (default: config.json).
- --output-dir PATH: Directory for output files (default: current directory).
- --output-format [json|md|html]: Output format (default: json).
- --dry-run: Run without real API calls (for testing).
- --lightweight: Disable rich/pandas for faster execution.
- --use-gabriel: Include the Gabriel agent in deliberation.
- --gabriel-model [openai|anthropic|xai]: Model for Gabriel (default: openai).
- --agent-melchior MODEL: Override model for Melchior.
- --agent-balthasar MODEL: Override model for Balthasar.
- --agent-casper MODEL: Override model for Casper.

[bold]Examples of Potential Use:[/bold]
- Analyze a business strategy:
  [cyan]python magi_system_v5.py --problem "How can we increase market share in Europe?" --output-format html[/cyan]
- Optimize a technical process:
  [cyan]python magi_system_v5.py --problem "How to reduce server downtime?" --dry-run --lightweight[/cyan]
- Explore ethical dilemmas with Gabriel:
  [cyan]python magi_system_v5.py --problem "What are the ethical implications of AI in healthcare?" --use-gabriel --gabriel-model anthropic[/cyan]

For more details, run with --help or consult README.md.
"""

# Initialize rich console for enhanced CLI output (optional, disabled in lightweight mode)
console = Console()

# Load environment variables from .env file
load_dotenv()

# Initialize global configurations early to avoid NameError
def initialize_early_config(config_file: str = "config.json", use_gabriel: bool = False, gabriel_model: str = "openai") -> Tuple[Dict, Dict]:
    """
    Loads and initializes configuration settings early in the script to avoid reference errors.

    Args:
        config_file (str): Path to the configuration file (default: "config.json").
        use_gabriel (bool): Whether to include the Gabriel agent (default: False).
        gabriel_model (str): API model for Gabriel (default: "openai").

    Returns:
        Tuple[Dict, Dict]: (Agents configuration, API settings).
    """
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error("Configuration file not found, using default settings")
        config = {
            "agents": {
                "Melchior": {
                    "model": "gpt-4o",
                    "api": "openai",
                    "role": "You are an analytical agent, providing precise, data-driven solutions focusing on technical optimization and strategic planning."
                },
                "Balthasar": {
                    "model": "claude-3-5-sonnet-20241022",
                    "api": "anthropic",
                    "role": "You are a nurturing agent, proposing solutions that prioritize human welfare, emotional stability, and safety."
                },
                "Casper": {
                    "model": "grok-3",
                    "api": "xai",
                    "role": "You are an intuitive agent, offering creative, value-driven solutions that reflect ethical considerations."
                }
            },
            "api_settings": {
                "retry_attempts": 3,
                "retry_min_wait": 1,
                "retry_max_wait": 5,
                "consensus_threshold": 0.7
            }
        }

    # Add Gabriel if requested
    if use_gabriel:
        gabriel_config = {
            "model": "gpt-4o-mini" if gabriel_model == "openai" else "claude-3-5-sonnet-20241022" if gabriel_model == "anthropic" else "grok-3",
            "api": gabriel_model,
            "role": "You are a cost-optimization agent, focusing on budget-friendly solutions with high ROI."
        }
        config["agents"]["Gabriel"] = gabriel_config
        logging.info(f"Added Gabriel agent with model {gabriel_config['model']} (API: {gabriel_model})")
    else:
        config["agents"].pop("Gabriel", None)
        logging.info("Gabriel agent not requested, proceeding without it.")

    return config["agents"], config["api_settings"]

# Initialize global variables early
AGENTS = {}
API_SETTINGS = {}
try:
    AGENTS, API_SETTINGS = initialize_early_config()
except Exception as e:
    logging.error(f"Early config initialization failed: {str(e)}")
    rprint(f"\n[yellow]An error occurred during initialization: {str(e)}[/yellow]")
    rprint(HELP_MESSAGE)
    sys.exit(1)

def validate_env_permissions(env_file: str = ".env") -> None:
    """
    Validates the permissions of the .env file to ensure security.

    Args:
        env_file (str): Path to the .env file (default: ".env").

    Raises:
        InsecureEnvFileError: If the file has insecure permissions (e.g., readable by group/others).
    """
    if not os.path.exists(env_file):
        logging.warning(f"{env_file} not found, assuming environment variables are set externally.")
        return
    
    # Check file permissions (should be 0600: readable/writable only by owner)
    file_stat = os.stat(env_file)
    permissions = stat.S_IMODE(file_stat.st_mode)
    if permissions & (stat.S_IRWXG | stat.S_IRWXO):
        error_msg = (
            f"Security warning: {env_file} has insecure permissions ({oct(permissions)}). "
            "It should be readable/writable only by the owner (chmod 600 .env)."
        )
        logging.error(error_msg)
        raise InsecureEnvFileError(error_msg)
    logging.info(f"{env_file} permissions validated successfully.")

def load_config(config_file: str = "config.json", use_gabriel: bool = False, gabriel_model: str = "openai") -> Dict:
    """
    Loads configuration settings from a JSON file and adjusts agents based on user input.

    Args:
        config_file (str): Path to the configuration file (default: "config.json").
        use_gabriel (bool): Whether to include the Gabriel agent (default: False).
        gabriel_model (str): API model for Gabriel (default: "openai").

    Returns:
        Dict: Configuration dictionary with agent settings and API parameters.
    """
    # Since we already initialized AGENTS and API_SETTINGS, reuse them
    config = {
        "agents": AGENTS,
        "api_settings": API_SETTINGS
    }
    return config

# Configuration setup with dynamic agent loading
def initialize_config(args) -> Dict:
    """
    Initializes the configuration based on command-line arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Dict: Loaded and adjusted configuration.
    """
    return load_config(args.config, args.use_gabriel, args.gabriel_model)

# Environment variables for API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")

def validate_api_keys() -> None:
    """
    Validates that all required API keys are present in the environment.

    Raises:
        MissingAPIKeyError: If any required API keys are missing.
    """
    required_keys = {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
        "XAI_API_KEY": XAI_API_KEY
    }
    missing_keys = [key for key, value in required_keys.items() if not value]
    if missing_keys:
        error_msg = f"Missing API keys: {', '.join(missing_keys)}. Please set them in the .env file or as environment variables."
        logging.error(error_msg)
        raise MissingAPIKeyError(error_msg)

# Semantic model for similarity checks between agent proposals
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Confidence log management for tracking historical confidence scores
def load_confidence_log(output_dir: str, problem: str, log_file: str = "confidence_log.json") -> List[float]:
    """
    Loads historical confidence scores from a log file located in analysis_results/<problem_folder>.

    Args:
        output_dir (str): Base directory for output files.
        problem (str): Problem statement to determine the subfolder.
        log_file (str): Name of the confidence log file (default: "confidence_log.json").

    Returns:
        List[float]: List of historical confidence scores.
    """
    problem_folder = create_problem_folder_name(problem)
    result_dir = os.path.join(output_dir, "analysis_results", problem_folder)
    log_path = os.path.join(result_dir, log_file)
    try:
        with open(log_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_confidence_log(confidences: List[float], output_dir: str, problem: str, log_file: str = "confidence_log.json") -> None:
    """
    Saves confidence scores to a log file in analysis_results/<problem_folder>, appending to historical data.

    Args:
        confidences (List[float]): List of confidence scores to save.
        output_dir (str): Base directory for output files.
        problem (str): Problem statement to determine the subfolder.
        log_file (str): Name of the confidence log file (default: "confidence_log.json").
    """
    problem_folder = create_problem_folder_name(problem)
    result_dir = os.path.join(output_dir, "analysis_results", problem_folder)
    os.makedirs(result_dir, exist_ok=True)
    log_path = os.path.join(result_dir, log_file)
    historical = load_confidence_log(output_dir, problem, log_file)
    historical.extend(confidences)
    with open(log_path, "w") as f:
        json.dump(historical, f)

# Custom xAI client for API interactions
class AsyncXAIClient:
    """
    Asynchronous client for interacting with the xAI API.
    """
    def __init__(self, api_key: str, base_url: str = "https://api.x.ai/v1"):
        """
        Initializes the xAI client with an API key and base URL.

        Args:
            api_key (str): xAI API key.
            base_url (str): Base URL for xAI API (default: "https://api.x.ai/v1").
        """
        self.api_key = api_key
        self.base_url = base_url

    @retry(
        stop=stop_after_attempt(API_SETTINGS["retry_attempts"]),
        wait=wait_exponential(multiplier=1, min=API_SETTINGS["retry_min_wait"], max=API_SETTINGS["retry_max_wait"]),
        retry=retry_if_exception_type(aiohttp.ClientError)
    )
    async def chat_completions_create(self, model: str, messages: List[Dict], dry_run: bool = False) -> Dict:
        """
        Makes an asynchronous API call to xAI's chat completions endpoint.

        Args:
            model (str): Model name to use (e.g., "grok-3").
            messages (List[Dict]): List of message dictionaries for the API call.
            dry_run (bool): If True, skips the API call and returns a mock response.

        Returns:
            Dict: API response containing the generated content.

        Raises:
            XAIError: If the API call fails after retries.
        """
        if dry_run:
            logging.info("Dry-run mode: Skipping xAI API call")
            return {"choices": [{"message": {"content": "Dry-run: No response generated"}}]}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": 0
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/chat/completions", headers=headers, json=payload) as response:
                response.raise_for_status()
                return await response.json()

# Initialize API clients for OpenAI, Anthropic, and xAI
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
xai_client = AsyncXAIClient(api_key=XAI_API_KEY)

async def call_openai(agent_name: str, problem: str, previous_proposals: List[Dict] = None, dry_run: bool = False) -> Dict:
    """
    Calls the OpenAI API for a given agent to generate a proposal.

    Args:
        agent_name (str): Name of the agent (e.g., "Melchior").
        problem (str): Problem statement for the agent to solve.
        previous_proposals (List[Dict]): List of previous proposals for context (optional).
        dry_run (bool): If True, skips the API call and returns a mock response.

    Returns:
        Dict: Proposal containing solution, justification, and confidence.

    Raises:
        OpenAIError: If the API call fails after retries.
    """
    try:
        if dry_run:
            logging.info(f"Dry-run mode: Skipping OpenAI call for {agent_name}")
            return {
                "agent": agent_name,
                "solution": "Dry-run: No response generated",
                "justification": "Dry-run: No justification provided",
                "confidence": 0.0
            }
        response = await openai_client.chat.completions.create(
            model=AGENTS[agent_name]["model"],
            messages=[
                {"role": "system", "content": AGENTS[agent_name]["role"]},
                {"role": "user", "content": f"Problem: {problem}\nPrevious proposals: {json.dumps(previous_proposals, default=str) if previous_proposals else 'None'}\nPropose a solution with a detailed justification and a confidence score (0-1). Format your response as:\nSolution: ...\nJustification: ...\nConfidence: X.XX"}
            ]
        )
        content = response.choices[0].message.content
        logging.info(f"OpenAI response for {agent_name}: {content[:100]}...")
        return parse_response(content, agent_name, raw_content=content)
    except Exception as e:
        logging.error(f"OpenAI API error for {agent_name}: {str(e)}")
        raise OpenAIError(f"OpenAI API error: {str(e)}")

async def call_anthropic(agent_name: str, problem: str, previous_proposals: List[Dict] = None, dry_run: bool = False) -> Dict:
    """
    Calls the Anthropic API for a given agent to generate a proposal.

    Args:
        agent_name (str): Name of the agent (e.g., "Balthasar").
        problem (str): Problem statement for the agent to solve.
        previous_proposals (List[Dict]): List of previous proposals for context (optional).
        dry_run (bool): If True, skips the API call and returns a mock response.

    Returns:
        Dict: Proposal containing solution, justification, and confidence.

    Raises:
        AnthropicError: If the API call fails after retries.
    """
    try:
        if dry_run:
            logging.info(f"Dry-run mode: Skipping Anthropic call for {agent_name}")
            return {
                "agent": agent_name,
                "solution": "Dry-run: No response generated",
                "justification": "Dry-run: No justification provided",
                "confidence": 0.0
            }
        response = await anthropic_client.messages.create(
            model=AGENTS[agent_name]["model"],
            max_tokens=1000,
            messages=[
                {"role": "user", "content": f"{AGENTS[agent_name]['role']}\nProblem: {problem}\nPrevious proposals: {json.dumps(previous_proposals, default=str) if previous_proposals else 'None'}\nPropose a solution with a detailed justification and a confidence score (0-1). Format your response as:\nSolution: ...\nJustification: ...\nConfidence: X.XX"}
            ]
        )
        content = response.content[0].text
        logging.info(f"Anthropic response for {agent_name}: {content[:100]}...")
        return parse_response(content, agent_name, raw_content=content)
    except Exception as e:
        logging.error(f"Anthropic API error for {agent_name}: {str(e)}")
        raise AnthropicError(f"Anthropic API error: {str(e)}")

async def call_xai(agent_name: str, problem: str, previous_proposals: List[Dict] = None, dry_run: bool = False) -> Dict:
    """
    Calls the xAI API for a given agent to generate a proposal.

    Args:
        agent_name (str): Name of the agent (e.g., "Casper").
        problem (str): Problem statement for the agent to solve.
        previous_proposals (List[Dict]): List of previous proposals for context (optional).
        dry_run (bool): If True, skips the API call and returns a mock response.

    Returns:
        Dict: Proposal containing solution, justification, and confidence.

    Raises:
        XAIError: If the API call fails after retries.
    """
    try:
        response = await xai_client.chat_completions_create(
            model=AGENTS[agent_name]["model"],
            messages=[
                {"role": "system", "content": AGENTS[agent_name]["role"]},
                {"role": "user", "content": f"Problem: {problem}\nPrevious proposals: {json.dumps(previous_proposals, default=str) if previous_proposals else 'None'}\nPropose a solution with a detailed justification and a confidence score (0-1). Format your response as:\nSolution: ...\nJustification: ...\nConfidence: X.XX"}
            ],
            dry_run=dry_run
        )
        content = response["choices"][0]["message"]["content"]
        logging.info(f"xAI response for {agent_name}: {content[:100]}...")
        return parse_response(content, agent_name, raw_content=content)
    except Exception as e:
        logging.error(f"xAI API error for {agent_name}: {str(e)}")
        raise XAIError(f"xAI API error: {str(e)}")

def parse_response(content: str, agent_name: str, raw_content: str = None) -> Dict:
    """
    Parses the API response into a structured proposal format.

    Args:
        content (str): Raw response content from the API.
        agent_name (str): Name of the agent that generated the response.
        raw_content (str): Original raw content for logging (optional).

    Returns:
        Dict: Parsed proposal with solution, justification, and confidence.
    """
    solution = ""
    justification = ""
    confidence = 0.0

    solution_pattern = r"Solution:\s*(.*?)(?=\nJustification:|\nConfidence:|$)"
    justification_pattern = r"Justification:\s*(.*?)(?=\nConfidence:|$)"
    confidence_pattern = r"Confidence:\s*(\d*\.?\d*)"

    solution_match = re.search(solution_pattern, content, re.DOTALL | re.IGNORECASE)
    if solution_match:
        solution = solution_match.group(1).strip()

    justification_match = re.search(justification_pattern, content, re.DOTALL | re.IGNORECASE)
    if justification_match:
        justification = justification_match.group(1).strip()

    confidence_match = re.search(confidence_pattern, content, re.IGNORECASE)
    if confidence_match:
        try:
            confidence = float(confidence_match.group(1))
            if not 0 <= confidence <= 1:
                confidence = 0.0
        except ValueError:
            confidence = 0.0

    if not solution or not justification:
        logging.warning(f"Failed to parse response for {agent_name}: {content[:200]}...")
        with open(f"raw_response_{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
            f.write(raw_content or content)
        solution = solution or "Parsing error: No solution found"
        justification = justification or "Parsing error: No justification found"

    return {
        "agent": agent_name,
        "solution": solution,
        "justification": justification,
        "confidence": confidence
    }

def check_consensus(proposals: List[Dict], output_dir: str, problem: str, threshold: float = API_SETTINGS["consensus_threshold"]) -> Tuple[bool, Dict]:
    """
    Checks if the proposals have reached consensus based on semantic similarity and confidence.

    Args:
        proposals (List[Dict]): List of agent proposals.
        output_dir (str): Base directory for output files.
        problem (str): Problem statement to determine the subfolder for logging.
        threshold (float): Consensus threshold (default: from config).

    Returns:
        Tuple[bool, Dict]: (Consensus achieved, Details of the consensus check).
    """
    solutions = [p["solution"] for p in proposals if p["solution"] and not p["solution"].startswith("Dry-run")]
    if len(solutions) < len(proposals):
        return False, {"reason": "Missing valid solutions from some agents"}

    embeddings = semantic_model.encode(solutions)
    similarity_scores = [
        util.cos_sim(embeddings[i], embeddings[j]).item()
        for i in range(len(solutions)) for j in range(i + 1, len(solutions))
    ]
    
    confidences = [p["confidence"] for p in proposals if not p["solution"].startswith("Dry-run")]
    historical_confidences = load_confidence_log(output_dir, problem)
    if historical_confidences:
        min_conf = min(historical_confidences)
        max_conf = max(historical_confidences)
        if max_conf > min_conf:
            confidences = [(c - min_conf) / (max_conf - min_conf) for c in confidences]
    
    avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    consensus_score = (avg_similarity * 0.8 + avg_confidence * 0.2)
    consensus = consensus_score >= threshold
    
    details = {
        "consensus_score": consensus_score,
        "semantic_similarity": avg_similarity,
        "avg_confidence": avg_confidence,
        "threshold": threshold,
        "individual_similarities": similarity_scores
    }
    logging.info(f"Consensus check: score={consensus_score}, consensus={consensus}, details={details}")
    
    save_confidence_log([p["confidence"] for p in proposals if not p["solution"].startswith("Dry-run")], output_dir, problem)
    return consensus, details

async def cross_justification_round(proposals: List[Dict], problem: str, agent_id_map: Dict[str, int], dry_run: bool = False) -> List[Dict]:
    """
    Conducts a cross-justification round where agents refine their proposals based on others' solutions.

    Args:
        proposals (List[Dict]): List of current agent proposals.
        problem (str): Original problem statement.
        agent_id_map (Dict[str, int]): Mapping of agent names to IDs.
        dry_run (bool): If True, skips API calls and uses mock responses.

    Returns:
        List[Dict]: List of refined proposals.
    """
    refined_proposals = []
    for agent_name, agent_id in agent_id_map.items():
        other_proposals = [
            {"id": idx, "solution": p["solution"][:100] + "...", "confidence": p["confidence"], "justification": p["justification"][:100] + "..."}
            for idx, p in enumerate(proposals) if p["agent"] != agent_name
        ]
        # Create the proposals string outside the f-string to avoid backslash issues in Python 3.10
        proposals_text = "\n".join([
            f"Agent {p['id']}: {p['solution']} (Confidence: {p['confidence']})\nJustification: {p['justification']}"
            for p in other_proposals
        ])
        cross_prompt = f"""
Original Problem: {problem}

Your previous proposal: {next(p['solution'] for p in proposals if p['agent'] == agent_name)}

Other agents' proposals (anonymized):
{proposals_text}

Review the other agents' solutions and justifications. Consider:
1. What strengths do you see in their approaches?
2. What concerns or weaknesses do you identify?
3. How might you refine your solution based on this analysis?
4. Do you maintain your original confidence level?

Provide your refined solution with updated justification and confidence.
Format your response as:
Solution: [your refined solution]
Justification: [your updated reasoning, addressing other proposals]
Confidence: [0.0-1.0]
"""
        try:
            refined_response = await globals()[f"call_{AGENTS[agent_name]['api']}"](agent_name, cross_prompt, dry_run=dry_run)
            refined_proposals.append(refined_response)
            logging.info(f"Cross-justification for Agent {agent_id}: {refined_response['solution'][:100]}...")
        except Exception as e:
            logging.error(f"Failed cross-justification for {agent_name}: {e}")
            original = next(p for p in proposals if p["agent"] == agent_name)
            refined_proposals.append(original)
            logging.info(f"Reused original proposal for {agent_name} due to error.")
    return refined_proposals

async def advanced_vote(proposals: List[Dict], problem: str, agent_id_map: Dict[str, int], dry_run: bool = False) -> Tuple[str, Dict, Dict]:
    """
    Conducts an advanced voting round to select the best proposal when consensus is not reached.

    Args:
        proposals (List[Dict]): List of agent proposals.
        problem (str): Original problem statement.
        agent_id_map (Dict[str, int]): Mapping of agent names to IDs.
        dry_run (bool): If True, skips API calls and uses mock responses.

    Returns:
        Tuple[str, Dict, Dict]: (Winner agent name, Votes dictionary, Voting details).
    """
    votes = {}
    justification_scores = {}
    
    for agent_name, agent_id in agent_id_map.items():
        anonymized_proposals = [
            {"id": idx, "solution": p["solution"], "confidence": p["confidence"], "justification": p["justification"]}
            for idx, p in enumerate(proposals)
        ]
        vote_prompt = f"""
You are Agent {agent_id}. Review these anonymized proposals for: {problem}

Proposals:
{json.dumps(anonymized_proposals, indent=2)}

Your task:
1. Vote for the BEST overall solution (excluding your own, Agent {agent_id_map[agent_name]})
2. Rate the quality of each justification (1-10 scale)
3. Consider: technical feasibility, human impact, ethical implications, practical implementation

Format your response as:
Vote: [agent id]
Vote_Justification: [why you chose this solution]
Justification_Scores: {', '.join([f'Agent {idx}={score}' for idx, score in enumerate([5]*len(proposals))])}
Overall_Reasoning: [your comprehensive analysis]
"""
        try:
            response = await globals()[f"call_{AGENTS[agent_name]['api']}"](agent_name, vote_prompt, dry_run=dry_run)
            vote_content = response.get("solution", "") + " " + response.get("justification", "")
            
            vote_match = re.search(r'Vote:\s*(\d+)', vote_content, re.IGNORECASE)
            voted_id = int(vote_match.group(1)) if vote_match else 0
            voted_agent = proposals[voted_id]["agent"] if voted_id < len(proposals) else list(AGENTS.keys())[0]
            
            scores = {}
            score_match = re.search(r'Justification_Scores:\s*(.+)', vote_content, re.IGNORECASE)
            if score_match:
                score_text = score_match.group(1)
                for idx, p in enumerate(proposals):
                    agent_score = re.search(rf'Agent {idx}=\d+', score_text, re.IGNORECASE)
                    scores[p["agent"]] = int(agent_score.group(0).split('=')[1]) if agent_score else 5
            else:
                scores = {p["agent"]: 5 for p in proposals}
            
            votes[agent_name] = {
                "vote": voted_agent,
                "vote_justification": response.get("justification", ""),
                "overall_reasoning": vote_content,
                "justification_scores": scores
            }
            justification_scores[agent_name] = scores
            logging.info(f"Vote by Agent {agent_id}: {voted_agent}, scores={scores}")
        except Exception as e:
            logging.error(f"Failed voting for {agent_name}: {e}")
            voted_agent = next(p["agent"] for p in proposals if p["agent"] != agent_name)
            votes[agent_name] = {
                "vote": voted_agent,
                "vote_justification": "Fallback vote due to error",
                "overall_reasoning": "Error occurred during voting",
                "justification_scores": {p["agent"]: 5 for p in proposals}
            }
            justification_scores[agent_name] = {p["agent"]: 5 for p in proposals}
    
    vote_counts = {}
    confidence_totals = {p["agent"]: p["confidence"] for p in proposals if not p["solution"].startswith("Dry-run")}
    justification_quality = {agent: 0 for agent in AGENTS}
    
    for agent_scores in justification_scores.values():
        for agent, score in agent_scores.items():
            justification_quality[agent] += score
    
    for vote_data in votes.values():
        voted_agent = vote_data["vote"]
        quality_bonus = justification_quality.get(voted_agent, 0) / 30
        weighted_vote = 1 + (quality_bonus * 0.2)
        vote_counts[voted_agent] = vote_counts.get(voted_agent, 0) + weighted_vote
    
    max_votes = max(vote_counts.values()) if vote_counts else 0
    winners = [agent for agent, count in vote_counts.items() if count == max_votes]
    
    if len(winners) > 1:
        winner_confidences = {w: confidence_totals.get(w, 0.0) for w in winners}
        max_confidence = max(winner_confidences.values())
        confidence_winners = [w for w, c in winner_confidences.items() if c == max_confidence]
        
        if len(confidence_winners) > 1:
            winner_quality = {w: justification_quality.get(w, 0) for w in confidence_winners}
            max_quality = max(winner_quality.values())
            quality_winners = [w for w, q in winner_quality.items() if q == max_quality]
            
            winner = list(AGENTS.keys())[0] if list(AGENTS.keys())[0] in quality_winners else quality_winners[0]
        else:
            winner = confidence_winners[0]
    else:
        winner = winners[0] if winners else list(AGENTS.keys())[0]
    
    voting_details = {
        "vote_counts": vote_counts,
        "justification_scores": justification_scores,
        "justification_quality_totals": justification_quality,
        "tiebreaker_used": len(winners) > 1
    }
    logging.info(f"Voting result: winner={winner}, details={voting_details}")
    
    return winner, votes, voting_details

def create_problem_folder_name(problem: str) -> str:
    """
    Creates a sanitized folder name from the problem statement.

    Args:
        problem (str): The problem statement.

    Returns:
        str: Sanitized folder name (lowercase, spaces replaced with underscores, special chars removed).
    """
    # Convert to lowercase, replace spaces with underscores, remove special characters
    sanitized = problem.lower().replace(" ", "_")
    sanitized = re.sub(r"[^a-z0-9_]", "", sanitized)
    # Truncate to avoid overly long folder names
    return sanitized[:50]

def setup_logging(output_dir: str, problem: str) -> None:
    """
    Sets up logging to write to a file in analysis_results/<problem_folder>.

    Args:
        output_dir (str): Base directory for output files.
        problem (str): Problem statement to determine the subfolder.
    """
    problem_folder = create_problem_folder_name(problem)
    result_dir = os.path.join(output_dir, "analysis_results", problem_folder)
    os.makedirs(result_dir, exist_ok=True)
    log_file = os.path.join(result_dir, "magi_system.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def save_output(output: Dict, output_dir: str, output_format: str, timestamp: str, problem: str, lightweight: bool = False) -> Tuple[str, str]:
    """
    Saves the deliberation output in the specified format under analysis_results/<problem_folder>.

    Args:
        output (Dict): Output dictionary containing deliberation results.
        output_dir (str): Base directory for output files.
        output_format (str): Format of the output ("json", "md", or "html").
        timestamp (str): Timestamp for the output filenames.
        problem (str): Problem statement to create a subfolder.
        lightweight (bool): If True, skips rich/pandas for faster execution.

    Returns:
        Tuple[str, str]: Base filename and votes filename for the saved outputs.
    """
    # Create the subfolder structure: analysis_results/<problem_folder>
    problem_folder = create_problem_folder_name(problem)
    result_dir = os.path.join(output_dir, "analysis_results", problem_folder)
    os.makedirs(result_dir, exist_ok=True)

    base_filename = os.path.join(result_dir, f"magi_output_{timestamp}")
    votes_filename = os.path.join(result_dir, f"magi_votes_{timestamp}")

    if output_format == "json":
        with open(f"{base_filename}.json", "w") as f:
            json.dump(output, f, indent=2)
        with open(f"{votes_filename}.json", "w") as f:
            json.dump(output["votes"], f, indent=2)
    elif output_format == "md":
        md_content = f"# MAGI Output\n\n**Problem**: {output['problem']}\n**Iterations**: {output['iterations']}\n**Consensus**: {output['consensus']}\n\n## Proposals\n\n"
        for p in output["proposals"]:
            md_content += f"- **Agent {p['agent']}**: {p['solution']}\n  - Justification: {p['justification']}\n  - Confidence: {p['confidence']}\n"
        md_content += "\n## Discussions\n\n"
        for agent, vote in output["votes"].items():
            md_content += f"- **Agent {agent}**: Voted for {vote['vote']}\n  - Justification: {vote['vote_justification']}\n"
        md_content += f"\n**Winner**: {output['winner']}\n**Final Solution**: {output['final_solution']}"
        with open(f"{base_filename}.md", "w") as f:
            f.write(md_content)
        with open(f"{votes_filename}.md", "w") as f:
            f.write("\n".join([f"- **Agent {a}**: Voted for {v['vote']}\n  - Justification: {v['vote_justification']}" for a, v in output["votes"].items()]))
    elif output_format == "html":
        if lightweight:
            html_content = f"""
            <html><head><style>body {{font-family: Arial;}} h1 {{color: #333;}}</style></head><body>
            <h1>MAGI Output</h1>
            <p><strong>Problem</strong>: {output['problem']}</p>
            <p><strong>Iterations</strong>: {output['iterations']}</p>
            <p><strong>Consensus</strong>: {output['consensus']}</p>
            <h2>Proposals</h2>
            <ul>
            {''.join([f"<li><strong>Agent {p['agent']}</strong>: {p['solution']}<br>Justification: {p['justification']}<br>Confidence: {p['confidence']}</li>" for p in output["proposals"]])}
            </ul>
            <h2>Discussions</h2>
            <ul>
            {''.join([f"<li><strong>Agent {a}</strong>: Voted for {v['vote']}<br>Justification: {v['vote_justification']}</li>" for a, v in output["votes"].items()])}
            </ul>
            <p><strong>Winner</strong>: {output['winner']}</p>
            <p><strong>Final Solution</strong>: {output['final_solution']}</p>
            </body></html>
            """
        else:
            df_proposals = pd.DataFrame(output["proposals"])
            df_votes = pd.DataFrame([(a, v["vote"], v["vote_justification"]) for a, v in output["votes"].items()], columns=["Agent", "Vote", "Justification"])
            consensus_details = output["consensus_details"]
            consensus_html = f"""
            <h2>Consensus Details</h2>
            <p><strong>Consensus Score</strong>: {consensus_details.get('consensus_score', 'N/A'):.3f}</p>
            <p><strong>Semantic Similarity</strong>: {consensus_details.get('semantic_similarity', 'N/A'):.3f}</p>
            <p><strong>Average Confidence</strong>: {consensus_details.get('avg_confidence', 'N/A'):.3f}</p>
            <p><strong>Threshold</strong>: {consensus_details.get('threshold', 'N/A'):.3f}</p>
            """
            html_content = f"""
            <html><head><style>table {{border-collapse: collapse; width: 100%;}} th, td {{border: 1px solid black; padding: 8px; text-align: left;}}</style></head><body>
            <h1>MAGI Output</h1>
            <p><strong>Problem</strong>: {output['problem']}</p>
            <p><strong>Iterations</strong>: {output['iterations']}</p>
            <p><strong>Consensus</strong>: {output['consensus']}</p>
            {consensus_html}
            <h2>Proposals</h2>
            {df_proposals.to_html(index=False)}
            <h2>Discussions</h2>
            {df_votes.to_html(index=False)}
            <p><strong>Winner</strong>: {output['winner']}</p>
            <p><strong>Final Solution</strong>: {output['final_solution']}</p>
            </body></html>
            """
        with open(f"{base_filename}.html", "w") as f:
            f.write(html_content)
        with open(f"{votes_filename}.html", "w") as f:
            f.write(html_content if lightweight else df_votes.to_html(index=False))

    return base_filename, votes_filename

async def main(problem: str, max_iterations: int, output_dir: str, output_format: str, dry_run: bool, agent_overrides: Dict, lightweight: bool = False) -> Dict:
    """
    Main entry point for the MAGI Decision Core v5 deliberation system.

    Args:
        problem (str): Problem statement for agents to solve.
        max_iterations (int): Maximum number of deliberation iterations.
        output_dir (str): Directory for saving output files.
        output_format (str): Output format ("json", "md", or "html").
        dry_run (bool): If True, skips API calls and uses mock responses.
        agent_overrides (Dict): Dictionary of agent model overrides.
        lightweight (bool): If True, skips rich/pandas for faster execution.

    Returns:
        Dict: Deliberation output dictionary.
    """
    # Validate .env file permissions for security
    validate_env_permissions()

    # Setup logging to the appropriate subfolder
    setup_logging(output_dir, problem)

    # Display welcome message and execution settings
    if not lightweight:
        console.print("\n[bold blue]=== MAGI Decision Core v5 ===[/bold blue]")
        console.print(f"[green]Problem:[/green] {problem}")
        console.print(f"[green]Dry-run mode:[/green] {'Enabled' if dry_run else 'Disabled'}")
        console.print(f"[green]Output format:[/green] {output_format.upper()}")
        console.print(f"[green]Max iterations:[/green] {max_iterations}")
        console.print(f"[green]Lightweight mode:[/green] {'Enabled' if lightweight else 'Disabled'}")
        console.print(f"[green]Agents:[/green] {', '.join(AGENTS.keys())}\n")
    else:
        print(f"=== MAGI Decision Core v5 ===")
        print(f"Problem: {problem}")
        print(f"Dry-run mode: {'Enabled' if dry_run else 'Disabled'}")
        print(f"Output format: {output_format.upper()}")
        print(f"Max iterations: {max_iterations}")
        print(f"Lightweight mode: Enabled")
        print(f"Agents: {', '.join(AGENTS.keys())}\n")

    # Apply agent overrides from command-line arguments
    for agent, model in agent_overrides.items():
        if agent in AGENTS:
            AGENTS[agent]["model"] = model

    # Create dynamic agent ID mapping based on loaded agents
    agent_id_map = {name: idx for idx, name in enumerate(AGENTS.keys())}

    proposals = []
    consensus = False
    iterations = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Round 1: Gather initial proposals from agents
    if not lightweight:
        console.print("[bold]Gathering initial proposals...[/bold]")
    else:
        print("Gathering initial proposals...")
    
    tasks = []
    for agent_name in AGENTS:
        if not lightweight:
            with Status(f"Processing {agent_name}...", console=console):
                task = globals()[f"call_{AGENTS[agent_name]['api']}"](agent_name, problem, dry_run=dry_run)
                tasks.append(task)
        else:
            print(f"Processing {agent_name}...")
            task = globals()[f"call_{AGENTS[agent_name]['api']}"](agent_name, problem, dry_run=dry_run)
            tasks.append(task)
    
    # Execute tasks with individual error handling
    results = []
    for task in asyncio.as_completed(tasks):
        try:
            result = await task
            results.append(result)
        except Exception as e:
            agent_name = list(AGENTS.keys())[len(results)]
            logging.error(f"Initial proposal failed for {agent_name}: {e}")
            results.append({
                "agent": agent_name,
                "solution": "Error: Failed to generate solution",
                "justification": f"Error: {str(e)}",
                "confidence": 0.0
            })
    proposals = results
    logging.info(f"Iteration 1: proposals={[{p['agent']: p['solution'][:50]} for p in proposals]}")
    
    # Display initial proposals in a table (or simple print in lightweight mode)
    if not lightweight:
        table = Table(title="Agent Proposals (Iteration 1)", show_lines=True)
        table.add_column("Agent", style="cyan")
        table.add_column("Solution", style="magenta")
        table.add_column("Confidence", style="green")
        for p in proposals:
            table.add_row(p["agent"], p["solution"][:100] + ("..." if len(p["solution"]) > 100 else ""), f"{p['confidence']:.2f}")
        console.print(table)
    else:
        print("Agent Proposals (Iteration 1):")
        for p in proposals:
            print(f"- {p['agent']}: {p['solution'][:100]}{'...' if len(p['solution']) > 100 else ''} (Confidence: {p['confidence']:.2f})")
    
    consensus, consensus_details = check_consensus(proposals, output_dir, problem)
    iterations += 1
    if not lightweight:
        console.print(f"\n[bold]Consensus Check:[/bold] {'[green]Reached[/green]' if consensus else '[red]Not reached[/red]'} (Score: {consensus_details['consensus_score']:.3f}, Threshold: {consensus_details['threshold']:.3f})")
    else:
        print(f"\nConsensus Check: {'Reached' if consensus else 'Not reached'} (Score: {consensus_details['consensus_score']:.3f}, Threshold: {consensus_details['threshold']:.3f})")

    # Round 2: Cross-justification if consensus not reached
    if not consensus and iterations < max_iterations:
        if not lightweight:
            console.print("\n[bold]Initiating cross-justification round...[/bold]")
        else:
            print("\nInitiating cross-justification round...")
        proposals = await cross_justification_round(proposals, problem, agent_id_map, dry_run=dry_run)
        consensus, consensus_details = check_consensus(proposals, output_dir, problem)
        iterations += 1
        if not lightweight:
            console.print(f"[bold]Consensus Check:[/bold] {'[green]Reached[/green]' if consensus else '[red]Not reached[/red]'} (Score: {consensus_details['consensus_score']:.3f}, Threshold: {consensus_details['threshold']:.3f})")
            table = Table(title=f"Agent Proposals (Iteration {iterations})", show_lines=True)
            table.add_column("Agent", style="cyan")
            table.add_column("Solution", style="magenta")
            table.add_column("Confidence", style="green")
            for p in proposals:
                table.add_row(p["agent"], p["solution"][:100] + ("..." if len(p["solution"]) > 100 else ""), f"{p['confidence']:.2f}")
            console.print(table)
        else:
            print(f"Consensus Check: {'Reached' if consensus else 'Not reached'} (Score: {consensus_details['consensus_score']:.3f}, Threshold: {consensus_details['threshold']:.3f})")
            print(f"Agent Proposals (Iteration {iterations}):")
            for p in proposals:
                print(f"- {p['agent']}: {p['solution'][:100]}{'...' if len(p['solution']) > 100 else ''} (Confidence: {p['confidence']:.2f})")

    # Round 3: Voting if consensus still not reached
    winner = None
    votes = {}
    voting_details = {}
    if not consensus:
        if not lightweight:
            console.print("\n[bold]Initiating voting round...[/bold]")
        else:
            print("\nInitiating voting round...")
        winner, votes, voting_details = await advanced_vote(proposals, problem, agent_id_map, dry_run=dry_run)
        if not lightweight:
            console.print(f"[bold]Voting Result:[/bold] Winner is [cyan]{winner}[/cyan]")
        else:
            print(f"Voting Result: Winner is {winner}")
    
    # Prepare and save output
    output = {
        "problem": problem,
        "iterations": iterations,
        "consensus": consensus,
        "consensus_details": consensus_details,
        "proposals": proposals,
        "votes": {k: {"vote": v["vote"], "vote_justification": v["vote_justification"], "overall_reasoning": v["overall_reasoning"], "justification_scores": v["justification_scores"]} for k, v in votes.items()},
        "voting_details": voting_details,
        "winner": winner if not consensus else None,
        "final_solution": next(p["solution"] for p in proposals if p["agent"] == winner) if not consensus else proposals[0]["solution"]
    }
    
    base_filename, votes_filename = save_output(output, output_dir, output_format, timestamp, problem, lightweight=lightweight)
    
    # Display final summary
    if not lightweight:
        console.print("\n[bold blue]=== Execution Complete ===[/bold blue]")
        console.print(f"[green]Final Solution:[/green] {output['final_solution'][:200]}{'...' if len(output['final_solution']) > 200 else ''}")
        console.print(f"[green]Consensus:[/green] {'Reached' if consensus else 'Not reached'}")
        console.print(f"[green]Iterations:[/green] {iterations}")
        console.print(f"[green]Output saved to:[/green] {os.path.abspath(base_filename)}.{output_format}")
        if not consensus:
            console.print("[yellow]Note:[/yellow] No consensus was reached. The potential solutions discussed are stored in the results file for your review.")
        if not consensus:
            console.print(f"[green]Winner:[/green] {winner}")
        console.print("\n")
    else:
        print("\n=== Execution Complete ===")
        print(f"Final Solution: {output['final_solution'][:200]}{'...' if len(output['final_solution']) > 200 else ''}")
        print(f"Consensus: {'Reached' if consensus else 'Not reached'}")
        print(f"Iterations: {iterations}")
        print(f"Output saved to: {os.path.abspath(base_filename)}.{output_format}")
        if not consensus:
            print("Note: No consensus was reached. The potential solutions discussed are stored in the results file for your review.")
        if not consensus:
            print(f"Winner: {winner}")
        print("\n")

    return output

if __name__ == "__main__":
    try:
        # Define the argument parser with a custom description
        parser = argparse.ArgumentParser(
            description="MAGI Decision Core v5: A multi-agent deliberation system.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=HELP_MESSAGE.replace("[bold blue]", "").replace("[/bold blue]", "")
                               .replace("[bold]", "").replace("[/bold]", "")
                               .replace("[green]", "").replace("[/green]", "")
                               .replace("[cyan]", "").replace("[/cyan]", "")
                               .replace("[red]", "").replace("[/red]", "")
        )
        parser.add_argument(
            "--problem",
            type=str,
            help="Problem statement for agents to deliberate (required)."
        )
        parser.add_argument(
            "--max-iterations",
            type=int,
            default=2,
            help="Maximum number of deliberation iterations (default: 2)."
        )
        parser.add_argument(
            "--config",
            type=str,
            default="config.json",
            help="Path to configuration file (default: config.json)."
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default=".",
            help="Directory for output files (default: current directory)."
        )
        parser.add_argument(
            "--output-format",
            choices=["json", "md", "html"],
            default="json",
            help="Output format: json, md (Markdown), or html (default: json)."
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Run without calling real APIs (uses placeholders)."
        )
        parser.add_argument(
            "--lightweight",
            action="store_true",
            help="Disable rich/pandas for faster execution (less verbose output)."
        )
        parser.add_argument(
            "--use-gabriel",
            action="store_true",
            help="Include the Gabriel agent in deliberation."
        )
        parser.add_argument(
            "--gabriel-model",
            type=str,
            choices=["openai", "anthropic", "xai"],
            default="openai",
            help="Model for Gabriel: openai, anthropic, or xai (default: openai)."
        )
        parser.add_argument(
            "--agent-melchior",
            type=str,
            help="Override model for Melchior (e.g., gpt-4o-mini)."
        )
        parser.add_argument(
            "--agent-balthasar",
            type=str,
            help="Override model for Balthasar (e.g., claude-3-opus)."
        )
        parser.add_argument(
            "--agent-casper",
            type=str,
            help="Override model for Casper (e.g., grok-3)."
        )

        # Parse command-line arguments
        args = parser.parse_args()

        # Check for missing problem statement
        if not args.problem or args.problem.strip() == "":
            rprint(HELP_MESSAGE)
            sys.exit(0)

        # Load configuration and apply agent overrides
        CONFIG = initialize_config(args)
        AGENTS = CONFIG["agents"]
        API_SETTINGS = CONFIG["api_settings"]
        
        agent_overrides = {}
        if args.agent_melchior:
            agent_overrides["Melchior"] = args.agent_melchior
        if args.agent_balthasar:
            agent_overrides["Balthasar"] = args.agent_balthasar
        if args.agent_casper:
            agent_overrides["Casper"] = args.agent_casper
        
        validate_api_keys()
        asyncio.run(main(
            args.problem,
            args.max_iterations,
            args.output_dir,
            args.output_format,
            args.dry_run,
            agent_overrides,
            args.lightweight
        ))
    except Exception as e:
        # Display friendly message on any execution error
        rprint(f"\n[yellow]An error occurred: {str(e)}[/yellow]")
        rprint(HELP_MESSAGE)
        sys.exit(1)
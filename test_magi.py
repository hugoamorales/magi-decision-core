"""
File: test_magi.py
Description: Unit tests for MAGI Decision Core v5.
Modifications:
- Added tests for new features (--lightweight, .env validation, dynamic agents).
- Added detailed docstrings and comments for internal documentation.
"""
import pytest
import asyncio
import os
from magi_system_v5 import parse_response, check_consensus, advanced_vote, validate_env_permissions, main

@pytest.mark.asyncio
async def test_parse_response():
    """
    Tests parsing of a well-formed API response.
    """
    content = "Solution: Test Solution\nJustification: Test Reason\nConfidence: 0.8"
    result = parse_response(content, "Melchior", content)
    assert result["solution"] == "Test Solution"
    assert result["justification"] == "Test Reason"
    assert result["confidence"] == 0.8

@pytest.mark.asyncio
async def test_parse_response_failure():
    """
    Tests parsing of an invalid API response.
    """
    content = "Invalid response"
    result = parse_response(content, "Melchior", content)
    assert result["solution"] == "Parsing error: No solution found"
    assert result["justification"] == "Parsing error: No justification found"
    assert result["confidence"] == 0.0

def test_consensus():
    """
    Tests consensus check with similar solutions.
    """
    proposals = [
        {"agent": "Melchior", "solution": "Solution A", "confidence": 0.8},
        {"agent": "Balthasar", "solution": "Solution A", "confidence": 0.9},
        {"agent": "Casper", "solution": "Solution A", "confidence": 0.85}
    ]
    consensus, details = check_consensus(proposals, threshold=0.7)
    assert consensus is True
    assert details["consensus_score"] > 0.7

@pytest.mark.asyncio
async def test_voting():
    """
    Tests the voting mechanism in dry-run mode.
    """
    proposals = [
        {"agent": "Melchior", "solution": "Solution A", "confidence": 0.8, "justification": "Reason A"},
        {"agent": "Balthasar", "solution": "Solution B", "confidence": 0.9, "justification": "Reason B"},
        {"agent": "Casper", "solution": "Solution C", "confidence": 0.85, "justification": "Reason C"}
    ]
    agent_id_map = {"Melchior": 0, "Balthasar": 1, "Casper": 2}
    
    async def mock_call_api(agent_name, prompt, dry_run=False):
        if dry_run:
            return {
                "agent": agent_name,
                "solution": "Dry-run: No solution generated",
                "justification": "Dry-run: No justification provided",
                "confidence": 0.0
            }
        return {
            "solution": f"Vote: {(agent_id_map[agent_name] + 1) % 3}\nVote_Justification: Test vote\nJustification_Scores: Agent 0=5, Agent 1=6, Agent 2=7\nOverall_Reasoning: Test analysis",
            "justification": "Test vote"
        }
    
    # Mock API calls
    globals()['call_openai'] = mock_call_api
    globals()['call_anthropic'] = mock_call_api
    globals()['call_xai'] = mock_call_api
    
    winner, votes, details = await advanced_vote(proposals, "Test Problem", agent_id_map, dry_run=True)
    assert winner in ["Melchior", "Balthasar", "Casper"]
    assert len(votes) == 3
    for agent in votes:
        assert votes[agent]["vote_justification"].startswith("Dry-run")

def test_validate_env_permissions(tmp_path):
    """
    Tests validation of .env file permissions.
    
    Args:
        tmp_path: Pytest fixture for temporary directory.
    """
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_KEY=value")
    
    # Set insecure permissions (readable by others)
    os.chmod(env_file, 0o644)
    with pytest.raises(Exception, match="insecure permissions"):
        validate_env_permissions(str(env_file))
    
    # Set secure permissions (readable/writable by owner only)
    os.chmod(env_file, 0o600)
    validate_env_permissions(str(env_file))  # Should not raise

@pytest.mark.asyncio
async def test_dynamic_agents():
    """
    Tests handling of dynamic agents from config.json.
    """
    proposals = [
        {"agent": "Melchior", "solution": "Solution A", "confidence": 0.8},
        {"agent": "Balthasar", "solution": "Solution A", "confidence": 0.9},
        {"agent": "Casper", "solution": "Solution A", "confidence": 0.85},
        {"agent": "Gabriel", "solution": "Solution A", "confidence": 0.7}
    ]
    consensus, details = check_consensus(proposals, threshold=0.7)
    assert consensus is True
    assert details["consensus_score"] > 0.7

@pytest.mark.asyncio
async def test_lightweight_mode():
    """
    Tests lightweight mode execution.
    """
    # Mock API calls
    async def mock_call_api(agent_name, problem, previous_proposals=None, dry_run=False):
        return {
            "agent": agent_name,
            "solution": "Mock solution",
            "justification": "Mock justification",
            "confidence": 0.8
        }
    globals()['call_openai'] = mock_call_api
    globals()['call_anthropic'] = mock_call_api
    globals()['call_xai'] = mock_call_api

    output = await main(
        problem="Test problem",
        max_iterations=1,
        output_dir=".",
        output_format="json",
        dry_run=True,
        agent_overrides={},
        lightweight=True
    )
    assert output["problem"] == "Test problem"
    assert "proposals" in output
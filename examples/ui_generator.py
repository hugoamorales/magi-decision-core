"""
MAGI Pipeline UI Generator
Creates interactive HTML visualizations for pipeline results.
"""

from datetime import datetime
from typing import Dict, Any, List


def generate_pipeline_ui(results: Dict[str, Any], ui_config: Dict[str, Any]) -> str:
    """Generate interactive HTML UI for pipeline results."""

    agent_colors = ui_config.get("agent_colors", {
        "Melchior": "#3498db",
        "Balthasar": "#e74c3c",
        "Casper": "#2ecc71"
    })

    pipeline_name = results.get("pipeline", "Pipeline Results")
    pipeline_type = results.get("pipeline_type", "unknown")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{pipeline_name} - Results</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}

        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 30px 40px;
        }}

        .header h1 {{
            font-size: 32px;
            margin-bottom: 10px;
        }}

        .header .meta {{
            opacity: 0.9;
            font-size: 14px;
        }}

        .inputs-section {{
            background: #f8f9fa;
            padding: 20px 40px;
            border-bottom: 1px solid #e0e0e0;
        }}

        .inputs-section h2 {{
            font-size: 18px;
            margin-bottom: 15px;
            color: #1e3c72;
        }}

        .input-item {{
            margin-bottom: 12px;
        }}

        .input-label {{
            font-weight: 600;
            color: #555;
            display: inline-block;
            min-width: 150px;
        }}

        .input-value {{
            color: #333;
        }}

        .progress-bar {{
            background: #e0e0e0;
            height: 8px;
            border-radius: 4px;
            margin: 20px 40px;
            overflow: hidden;
        }}

        .progress-fill {{
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            height: 100%;
            transition: width 0.3s ease;
        }}

        .stages-container {{
            padding: 20px 40px;
        }}

        .stage {{
            margin-bottom: 40px;
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            overflow: hidden;
            background: white;
        }}

        .stage-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .stage-header h3 {{
            font-size: 20px;
            font-weight: 600;
        }}

        .stage-header .stage-status {{
            background: rgba(255,255,255,0.2);
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }}

        .stage-header .stage-status.completed {{
            background: rgba(46, 204, 113, 0.9);
        }}

        .stage-description {{
            padding: 15px 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #e0e0e0;
            font-style: italic;
            color: #666;
        }}

        .consensus-info {{
            padding: 15px 20px;
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            margin: 20px;
        }}

        .consensus-info.reached {{
            background: #d4edda;
            border-left-color: #28a745;
        }}

        .proposals-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            padding: 20px;
        }}

        .proposal-card {{
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
            background: white;
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .proposal-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        }}

        .proposal-header {{
            padding: 15px 20px;
            color: white;
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .proposal-header .agent-name {{
            font-size: 18px;
        }}

        .confidence-badge {{
            background: rgba(255,255,255,0.3);
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
        }}

        .confidence-meter {{
            height: 4px;
            background: rgba(255,255,255,0.3);
            margin-top: 8px;
            border-radius: 2px;
            overflow: hidden;
        }}

        .confidence-fill {{
            height: 100%;
            background: white;
            transition: width 0.3s ease;
        }}

        .proposal-content {{
            padding: 20px;
            line-height: 1.6;
        }}

        .proposal-content h4 {{
            margin-top: 15px;
            margin-bottom: 8px;
            color: #1e3c72;
            font-size: 16px;
        }}

        .proposal-content p {{
            margin-bottom: 12px;
            color: #444;
        }}

        .proposal-content pre {{
            background: #f8f9fa;
            padding: 12px;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 14px;
        }}

        .justification {{
            margin-top: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            border-radius: 4px;
        }}

        .justification-label {{
            font-weight: 600;
            color: #667eea;
            margin-bottom: 8px;
        }}

        .footer {{
            background: #f8f9fa;
            padding: 20px 40px;
            text-align: center;
            color: #666;
            font-size: 14px;
        }}

        .toggle-icon {{
            transition: transform 0.3s ease;
        }}

        .toggle-icon.collapsed {{
            transform: rotate(-90deg);
        }}

        .stage-content {{
            max-height: 5000px;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }}

        .stage-content.collapsed {{
            max-height: 0;
        }}

        @media (max-width: 768px) {{
            .proposals-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 {pipeline_name}</h1>
            <div class="meta">
                <span>Started: {results.get('start_time', 'N/A')}</span> |
                <span>Duration: {results.get('duration_seconds', 0):.2f}s</span> |
                <span>Stages: {len(results.get('stages', []))}</span>
            </div>
        </div>
"""

    # Inputs section
    if results.get("inputs"):
        html += generate_inputs_section(results["inputs"])

    # Progress bar
    total_stages = len(results.get("stages", []))
    html += f"""
        <div class="progress-bar">
            <div class="progress-fill" style="width: 100%;"></div>
        </div>
"""

    # Stages
    html += '<div class="stages-container">\n'

    for idx, stage in enumerate(results.get("stages", []), 1):
        html += generate_stage_section(stage, idx, agent_colors)

    html += '</div>\n'

    # Footer
    html += f"""
        <div class="footer">
            Generated by MAGI Decision Core | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>

    <script>
        // Toggle stage collapse/expand
        document.querySelectorAll('.stage-header').forEach(header => {{
            header.addEventListener('click', () => {{
                const content = header.nextElementSibling;
                while (content && !content.classList.contains('stage-content')) {{
                    content = content.nextElementSibling;
                }}
                if (content) {{
                    content.classList.toggle('collapsed');
                    const icon = header.querySelector('.toggle-icon');
                    if (icon) {{
                        icon.classList.toggle('collapsed');
                    }}
                }}
            }});
        }});

        // Smooth scroll animations
        document.querySelectorAll('.proposal-card').forEach((card, idx) => {{
            card.style.animation = `fadeIn 0.5s ease ${{idx * 0.1}}s both`;
        }});

        const style = document.createElement('style');
        style.textContent = `
            @keyframes fadeIn {{
                from {{
                    opacity: 0;
                    transform: translateY(20px);
                }}
                to {{
                    opacity: 1;
                    transform: translateY(0);
                }}
            }}
        `;
        document.head.appendChild(style);
    </script>
</body>
</html>
"""

    return html


def generate_inputs_section(inputs: Dict[str, Any]) -> str:
    """Generate the inputs section of the UI."""
    html = '<div class="inputs-section">\n'
    html += '<h2>📝 Input Parameters</h2>\n'

    for key, value in inputs.items():
        # Format key nicely
        label = key.replace('_', ' ').title()
        html += f'<div class="input-item">\n'
        html += f'<span class="input-label">{label}:</span>\n'
        html += f'<span class="input-value">{value}</span>\n'
        html += '</div>\n'

    html += '</div>\n'
    return html


def generate_stage_section(stage: Dict[str, Any], stage_num: int, agent_colors: Dict[str, str]) -> str:
    """Generate a single stage section."""
    stage_name = stage.get("stage", f"Stage {stage_num}")
    description = stage.get("description", "")

    html = f'<div class="stage">\n'
    html += f'<div class="stage-header">\n'
    html += f'<div>\n'
    html += f'<h3><span class="toggle-icon">▼</span> Stage {stage_num}: {stage_name}</h3>\n'
    html += f'</div>\n'
    html += f'<span class="stage-status completed">✓ Completed</span>\n'
    html += f'</div>\n'

    # Wrap all content in stage-content for collapsing
    html += '<div class="stage-content">\n'

    if description:
        html += f'<div class="stage-description">{description}</div>\n'

    # Consensus info
    if "consensus" in stage:
        consensus = stage["consensus"]
        details = stage.get("details", {})
        consensus_class = "reached" if consensus else ""
        consensus_text = "✓ Consensus Reached" if consensus else "✗ Consensus Not Reached"
        score = details.get("consensus_score", 0)

        html += f'<div class="consensus-info {consensus_class}">\n'
        html += f'<strong>{consensus_text}</strong>\n'
        html += f'<br>Consensus Score: {score:.3f}\n'
        html += f'</div>\n'

    # Proposals
    proposals = stage.get("proposals", [])
    if proposals:
        html += '<div class="proposals-grid">\n'

        for proposal in proposals:
            agent_name = proposal.get("agent", "Unknown")
            agent_color = agent_colors.get(agent_name, "#95a5a6")
            confidence = proposal.get("confidence", 0.0)
            solution = proposal.get("solution", "")
            response = proposal.get("response", "")
            justification = proposal.get("justification", "")

            # Use solution if available, otherwise use response
            content = solution if solution else response

            html += f'<div class="proposal-card">\n'
            html += f'<div class="proposal-header" style="background-color: {agent_color};">\n'
            html += f'<span class="agent-name">{agent_name}</span>\n'

            if confidence > 0:
                html += f'<span class="confidence-badge">Confidence: {confidence:.0%}</span>\n'

            html += '</div>\n'

            if confidence > 0:
                html += '<div class="confidence-meter">\n'
                html += f'<div class="confidence-fill" style="width: {confidence*100}%;"></div>\n'
                html += '</div>\n'

            html += '<div class="proposal-content">\n'

            # Format content with basic parsing
            formatted_content = format_proposal_content(content)
            html += formatted_content

            if justification:
                html += '<div class="justification">\n'
                html += '<div class="justification-label">Justification:</div>\n'
                html += f'<div>{justification}</div>\n'
                html += '</div>\n'

            html += '</div>\n'  # proposal-content
            html += '</div>\n'  # proposal-card

        html += '</div>\n'  # proposals-grid

    html += '</div>\n'  # stage-content
    html += '</div>\n'  # stage

    return html


def format_proposal_content(content: str) -> str:
    """Format proposal content with basic HTML structure."""
    if not content:
        return "<p><em>No content</em></p>"

    # Simple formatting: preserve line breaks and detect sections
    lines = content.split('\n')
    formatted = []
    in_list = False

    for line in lines:
        line = line.strip()
        if not line:
            if in_list:
                formatted.append('</ul>')
                in_list = False
            continue

        # Detect headings (lines ending with :)
        if line.endswith(':') and len(line) < 100:
            if in_list:
                formatted.append('</ul>')
                in_list = False
            formatted.append(f'<h4>{line[:-1]}</h4>')

        # Detect list items
        elif line.startswith(('- ', '* ', '• ', '1. ', '2. ', '3. ', '4. ', '5. ')):
            if not in_list:
                formatted.append('<ul>')
                in_list = True
            # Remove list marker
            for prefix in ['- ', '* ', '• ', '1. ', '2. ', '3. ', '4. ', '5. ']:
                if line.startswith(prefix):
                    line = line[len(prefix):]
                    break
            formatted.append(f'<li>{line}</li>')

        # Regular paragraph
        else:
            if in_list:
                formatted.append('</ul>')
                in_list = False
            formatted.append(f'<p>{line}</p>')

    if in_list:
        formatted.append('</ul>')

    return '\n'.join(formatted)

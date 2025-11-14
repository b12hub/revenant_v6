import json
from pathlib import Path

def generate_architecture_docs(registry_path: str):
    """Generate Markdown architecture files for each agent series."""
    with open(registry_path, 'r', encoding='utf-8') as f:
        agents = json.load(f)

    grouped = {}
    for agent in agents:
        series = agent.get("series", "unknown").lower()
        grouped.setdefault(series, []).append(agent)

    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)

    for series, agent_list in grouped.items():
        md_path = docs_dir / f"{series}_architecture.md"
        with open(md_path, 'w', encoding='utf-8') as md:
            md.write(f"# {series.upper()} Architecture — Revenant Agent Series\n\n")
            md.write("## Purpose\n")
            md.write(f"Auto-generated documentation for {series.upper()} layer.\n\n")

            md.write("## Agents\n")
            for agent in agent_list:
                md.write(f"- **{agent['name']}** — {agent.get('description', 'No description')}\n")

            md.write("\n## Integration\n")
            md.write("This series integrates with adjacent layers based on dependency and function flow.\n\n")

            md.write("## Technical Notes\n")
            md.write("- Auto-generated from registry.json\n")
            md.write("- Version sync: on registry rebuild\n")

    print("✅ Architecture documentation successfully regenerated.")

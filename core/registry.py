"""
Revenant Agent Registry Builder
Scans and builds a registry of all RevenantAgentBase subclasses
"""

import json
import logging
import importlib.util
import inspect
from datetime import datetime
import sys
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')
logger = logging.getLogger('registry')


class RegistryBuilder:
    """Builds and maintains the agent registry"""

    def __init__(self, agents_path: str = "agents", output_path: str = "docs/registry.json"):
        self.agents_path = Path(agents_path)
        self.output_path = Path(output_path)
        self.agents_found: List[Dict[str, Any]] = []

    def load_agents(self):
        agents = []
        for agent_file in self.agents_path.glob("*.py"):
            if agent_file.name.startswith("__"):
                continue
            module_name = f"agents.{agent_file.stem}"
            module = importlib.import_module(module_name)
            for attr in dir(module):
                obj = getattr(module, attr)
                if (
                        hasattr(obj, "run")
                        and hasattr(obj, "name")
                        and getattr(obj, "name", None) != "RevenantAgentBase"
                ):
                    agents.append(obj)
        return agents

    def scan_agents_directory(self) -> None:
        """Recursively scan all agents, supporting nested series directories (a/b/c/d, etc.)"""
        logger.info(f"Recursively scanning {self.agents_path} for agents...")

        if not self.agents_path.exists():
            logger.warning(f"Agents directory {self.agents_path} not found")
            return

        sys.path.insert(0, str(Path('.').resolve()))  # ensure project root importable

        imported_modules = 0

        for file_path in self.agents_path.rglob("*.py"):
            if file_path.name == "__init__.py__":
                continue

            # Build module name like agents.a_series.agent_x
            relative_path = file_path.relative_to(self.agents_path).with_suffix("")
            module_name = f"agents.{'.'.join(relative_path.parts)}"

            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    imported_modules += 1
                    self._find_agent_classes(module, module_name)

            except Exception as e:
                logger.warning(f"Failed to import {file_path}: {e}")
                continue

        logger.info(f"Imported {imported_modules} modules, found {len(self.agents_found)} agents")




    def _find_agent_classes(self, module: Any, module_name: str) -> None:
        """Find all RevenantAgentBase subclasses in a module"""
        for name, obj in inspect.getmembers(module):
            if (
                    inspect.isclass(obj)
                    and not name.startswith('_')
                    and self._is_agent_subclass(obj)
                    and obj.__name__ != "RevenantAgentBase"
            ):
                # Extract metadata
                metadata = self._extract_metadata(obj, module_name)
                if metadata:
                    self.agents_found.append(metadata)
                    logger.info(f"Found agent: {metadata['name']} (v{metadata['version']})")

    def _extract_metadata(self, agent_class: Any, module_name: str) -> Dict[str, Any]:
        """Extract metadata from agent class, automatically infer series"""
        metadata = {}

        try:
            if hasattr(agent_class, 'metadata') and isinstance(agent_class.metadata, dict):
                metadata = agent_class.metadata.copy()
            else:
                metadata = {
                    "name": agent_class.__name__,
                    "description": f"{agent_class.__name__} agent",
                }

            # Auto-detect series from module name (e.g., agents.a_series.agent_x)
            parts = module_name.split('.')
            series = next((p for p in parts if p.endswith('_series')), "unknown_series")

            metadata.setdefault("series", series)
            metadata.setdefault("version", "0.0.0")
            metadata.setdefault("module", module_name)

        except Exception as e:
            logger.warning(f"Failed to extract metadata from {agent_class.__name__}: {e}")
            return {}

        return metadata
    def _is_agent_subclass(self, cls: Any) -> bool:
        """Check if class is a subclass of RevenantAgentBase"""
        try:
            # Check class hierarchy for RevenantAgentBase
            for base in cls.__mro__:
                if base.__name__ == 'RevenantAgentBase':
                    return True
        except Exception:
            pass
        return False

    def build_registry(self):
        """Build the complete registry and write to file"""
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            # Standard registry
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(self.agents_found, f, indent=4)
            logger.info(f"Registry written to {self.output_path}")

            # Timestamped backup
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            backup_path = self.output_path.with_name(f"registry_{timestamp}.json")
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(self.agents_found, f, indent=4)
            logger.info(f"Backup created: {backup_path}")

            # Auto-generate architecture documentation
            from core.scripts.generate_architecture_docs import generate_architecture_docs
            generate_architecture_docs(self.output_path)
            logger.info("✅ Architecture docs updated successfully")

            return True

        except Exception as e:
            logger.warning(f"Failed to build registry: {e}")
            return False


def main():
    """Main entry point for registry builder"""
    builder = RegistryBuilder()
    builder.scan_agents_directory()
    success = builder.build_registry()

    if success:
        logger.info("Registry build completed successfully")
    else:
        logger.error("Registry build failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
"""Make sure that black and ruff versions used with nbQA match the primary versions.

For example, we could use ruff==v0.4.6 in the primary pre-commit config, but nbQA could
be using ruff==v0.3.2. In this case, we raise an error.

"""

from pathlib import Path
from typing import NotRequired, TypedDict

import yaml


class PreCommitRepo(TypedDict):
    repo: str
    hooks: list[dict]
    rev: NotRequired[str]


class PreCommitConfig(TypedDict):
    repos: list[PreCommitRepo]
    ci: dict


class NbQAHook(TypedDict):
    id: str
    additional_dependencies: list[str]


class NbQARepo(PreCommitRepo):
    hooks: list[NbQAHook]
    rev: str


PRE_COMMIT_CONFIG_FILE = Path(".pre-commit-config.yaml")


TOOL_TO_REPO = {
    "black": "https://github.com/psf/black",
    "ruff": "https://github.com/astral-sh/ruff-pre-commit",
    "nbQA": "https://github.com/nbQA-dev/nbQA",
}


def read_yaml(file: Path) -> PreCommitConfig:
    """Read a YAML file."""
    with file.open() as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as error:
            raise ValueError("Failed to parse .pre-commit-config.yaml file") from error


def get_nbqa_repo(pre_commit_config: PreCommitConfig) -> NbQARepo | None:
    """Get the nbQA repo from the pre-commit config.

    Args:
        pre_commit_config: The pre-commit config.

    Returns:
        The nbQA repo if found, otherwise None.

    """
    for repo in pre_commit_config["repos"]:
        if repo["repo"] == TOOL_TO_REPO["nbQA"]:
            return repo


def get_primary_version(pre_commit_config: PreCommitConfig, tool: str) -> str | None:
    """Get the primary version of the tool used in the pre-commit config.

    Args:
        pre_commit_config: The pre-commit config.
        tool: The tool to get the primary version for. For example, ruff or black.

    Returns:
        The primary version of the tool, otherwise None.

    """
    for repo in pre_commit_config["repos"]:
        if repo["repo"] == TOOL_TO_REPO[tool]:
            return repo["rev"]


def check_for_version_mismatch(
    pre_commit_config: PreCommitConfig,
    nbqa_repo: NbQARepo,
) -> None:
    """Check for version mismatch between primary versions and versions used by nbQA.

    Args:
        pre_commit_config: The pre-commit config.
        nbqa_repo: The nbQA repo.

    Raises:
        ValueError: If there is a version mismatch.

    """
    version_mismatch = {}

    for hook in nbqa_repo["hooks"]:
        tool = hook["id"].removeprefix("nbqa-")
        primary_version = get_primary_version(pre_commit_config, tool)
        tool_dependency = hook["additional_dependencies"][0]
        version_used_by_nbqa = tool_dependency.removeprefix(f"{tool}==")

        if primary_version != version_used_by_nbqa:
            version_mismatch[tool] = {
                "primary": primary_version,
                "nbQA": version_used_by_nbqa,
            }

    if version_mismatch:
        raise ValueError(f"Versions mismatch in nbQA repo: {version_mismatch}")


if __name__ == "__main__":
    pre_commit_config = read_yaml(PRE_COMMIT_CONFIG_FILE)
    nbqa_repo = get_nbqa_repo(pre_commit_config)
    check_for_version_mismatch(pre_commit_config, nbqa_repo)

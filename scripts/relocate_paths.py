"""Relocate all experiment YAML paths to this checkout."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml


def environment_pythons():
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda:
        for parent in Path(sys.executable).absolute().parents:
            candidate = parent / "bin/conda"
            if candidate.is_file():
                conda = str(candidate)
                break
    if not conda:
        raise ValueError("Activate a Conda environment before relocating paths")
    envs = json.loads(subprocess.check_output([conda, "env", "list", "--json"], text=True))["envs"]
    found = {name: [str(Path(env) / "bin/python") for env in envs
                    if Path(env).name == name and (Path(env) / "bin/python").is_file()]
             for name in ("libero_test", "lerobot_pi05")}
    for name, matches in found.items():
        if len(matches) != 1:
            raise ValueError(f"Expected one {name} environment, found {len(matches)}")
    return dict(simulator_python=found["libero_test"][0], evaluation_python=found["lerobot_pi05"][0])


def relocate(text, old, new, interpreters):
    """Rewrite path values while preserving YAML comments and formatting."""
    replacements = []

    def visit(node, key=None):
        if isinstance(node, yaml.MappingNode):
            for name, item in node.value:
                visit(item, name.value)
        elif isinstance(node, yaml.SequenceNode):
            for item in node.value:
                visit(item)
        elif node.tag == "tag:yaml.org,2002:str":
            path = Path(node.value)
            if key in interpreters:
                value = interpreters[key]
            elif path.is_absolute() and path.is_relative_to(old):
                value = str(new / path.relative_to(old))
            else:
                return
            if value != node.value:
                replacements.append((node.start_mark.index, node.end_mark.index, json.dumps(value, ensure_ascii=False)))

    visit(yaml.compose(text, Loader=yaml.SafeLoader))
    for start, end, value in sorted(set(replacements), reverse=True):
        text = text[:start] + value + text[end:]
    yaml.safe_load(text)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    configs = root / "configs"
    reference = configs / "pi05_holoq.yaml"
    settings = sorted(configs.glob("*.yaml"))
    if not reference.is_file() or not (root / "scripts/train.py").is_file():
        parser.error(f"Expected a PIVOT-Q-4090 checkout: {root}")
    paths = settings + [configs / "libero/config.yaml", configs / "libero_plus/config.yaml"]
    try:
        docs = {path: path.read_text() for path in paths}
        old = Path(yaml.safe_load(docs[reference])["benchmark"]["task_file"]).parent.parent
        if not old.is_absolute() or old == Path("/"):
            raise ValueError("Invalid old project root in benchmark.task_file")
        interpreters = environment_pythons()
        updates = {}
        for path, document in docs.items():
            relocated = relocate(document, old, root, interpreters)
            if relocated != document:
                updates[path] = relocated
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError, yaml.YAMLError) as error:
        parser.error(str(error))
    for path, contents in updates.items():
        print(f"{'Would update' if args.dry_run else 'Updating'} {path}")
        if not args.dry_run:
            temp = path.with_suffix(".yaml.tmp")
            temp.write_text(contents)
            temp.replace(path)
    print(f"{len(updates)}/{len(paths)} YAMLs changed; interpreters: {interpreters}")


if __name__ == "__main__":
    main()

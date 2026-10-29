import argparse
import os
from pathlib import Path
import yaml


def _parse_bool(value):
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false, yes/no, or 1/0")


def add_config_arguments(parser, flat_config):
    """Register YAML override arguments while preserving types, defaults, and authorization semantics of explicit CLI flags."""
    destinations = {action.dest for action in parser._actions}
    options = {option for action in parser._actions for option in action.option_strings}
    for key, value in flat_config.items():
        option = "--" + key
        if key in destinations or option in options:
            continue
        example = value[0] if isinstance(value, list) and value else value
        value_type = (_parse_bool if isinstance(example, bool) else
                      str if example is None or isinstance(example, list) else type(example))
        kwargs = {"nargs": "+"} if isinstance(value, list) else {}
        parser.add_argument(option, default=value, type=value_type, **kwargs)


def resolve_api_key(client_config, _seen=None):
    """Resolve local API credentials without copying secrets into generated snapshots.

    Explicit environment variable overrides preserve legacy client behavior;
    generated configs may reference a dataset YAML to read its active credential."""
    cfg = client_config or {}
    environment_name = cfg.get("api_key_env") or "LLM_API_KEY"
    environment_key = os.environ.get(environment_name)
    configured = cfg.get("api_key")
    key = environment_key or (os.path.expandvars(configured) if isinstance(configured, str) else None)
    if isinstance(key, str) and key.strip() and not key.startswith("$"):
        return key
    reference = cfg.get("api_key_config_file")
    if reference:
        path = Path(reference).expanduser().resolve()
        seen = set() if _seen is None else set(_seen)
        if path in seen:
            raise ValueError("Circular API credential configuration reference")
        seen.add(path)
        try:
            source = yaml_config_hook(path)
        except yaml.YAMLError:
            raise ValueError("API credential source must be valid YAML") from None
        if not isinstance(source, dict) or not isinstance(source.get("llm_client"), dict):
            raise ValueError("API credential source must contain llm_client")
        nested = dict(source["llm_client"])
        if nested.get("api_key_config_file"):
            child = Path(nested["api_key_config_file"])
            if not child.is_absolute():
                nested["api_key_config_file"] = str(path.parent / child)
        return resolve_api_key(nested, seen)
    raise ValueError("llm_client.api_key is unresolved; set it in the dataset YAML or LLM_API_KEY")


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def unflatten_dict(d, sep='.'):
    result_dict = {}
    for k, v in d.items():
        keys = k.split(sep)
        d = result_dict
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = v
    return result_dict

def yaml_config_hook(config_file):
    """Load a YAML configuration file with support for nested YAML includes.

    Provides a structured configuration entry point to reduce reliance on command-line flags."""

    # Load and merge nested YAML files specified under the defaults section.
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
        for d in cfg.get("defaults", []):
            if not isinstance(d, dict) or not d:
                continue
            config_dir, cf = next(iter(d.items()))
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf, "r", encoding="utf-8") as sub_f:
                l = yaml.safe_load(sub_f) or {}
                if isinstance(l, dict):
                    cfg.update(l)

    if "defaults" in cfg:
        del cfg["defaults"]

    return cfg

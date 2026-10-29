"""Explicit cross-provider Chat request options; never overrides messages, model, token limits, or request budgets."""

import re


_STRICT_MODEL_BASES = {
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
}


def normalize_request_options(options):
    """Validate and copy whitelisted fields; null for temperature/seed omits the field from the request."""
    if options is None:
        return {}
    allowed = {"temperature", "seed", "reasoning_effort", "max_tokens_parameter", "enable_thinking"}
    if not isinstance(options, dict) or set(options) - allowed:
        raise ValueError("request_options must contain only supported Chat request fields")
    value = options.get("temperature")
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                              or not 0 <= value <= 2):
        raise ValueError("request_options.temperature must be null or a finite number in [0, 2]")
    value = options.get("seed")
    if value is not None and type(value) is not int:
        raise ValueError("request_options.seed must be null or an integer")
    if "enable_thinking" in options and type(options["enable_thinking"]) is not bool:
        raise ValueError("request_options.enable_thinking must be a boolean")
    if "reasoning_effort" in options and options["reasoning_effort"] not in (
            "none", "minimal", "low", "medium", "high", "xhigh", "max"):
        raise ValueError("Unsupported request_options.reasoning_effort")
    if "max_tokens_parameter" in options and options["max_tokens_parameter"] not in (
            "max_tokens", "max_completion_tokens"):
        raise ValueError("max_tokens_parameter must be max_tokens or max_completion_tokens")
    return dict(options)


def normalize_response_model_names(names):
    """Additional accepted response model names must be declared explicitly rather than inferred from provider routing names."""
    if names is None:
        return []
    if not isinstance(names, list) or any(
            not isinstance(name, str) or not name.strip()
            or any(character.isspace() for character in name.strip()) for name in names):
        raise ValueError("response_model_names must be a list of nonempty model names")
    return list(dict.fromkeys(name.strip() for name in names))


def model_response_matches_request(requested_model: str, response_model: str,
                                   response_model_names=None) -> bool:
    """Check whether an API response belongs to the requested model family."""
    if response_model_names is not None:
        return any(model_response_matches_request(name, response_model) for name in
                   [requested_model, *normalize_response_model_names(response_model_names)])
    requested = str(requested_model or "").strip().lower().replace("_", ".")
    response = str(response_model or "").strip().lower().replace("_", ".")
    if not requested or not response:
        return False
    if requested in _STRICT_MODEL_BASES:
        return response == requested or re.fullmatch(
            re.escape(requested) + r"-[0-9]{4}-[0-9]{2}-[0-9]{2}", response
        ) is not None
    if any(requested.startswith(base + "-") for base in _STRICT_MODEL_BASES):
        return response == requested
    return response == requested or response.startswith(f"{requested}-")


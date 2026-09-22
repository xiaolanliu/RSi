"""Project-scoped provider selection; never reads or mutates Codex settings."""
import os
from pathlib import Path
import re
from urllib.parse import urlsplit


def read_credential(key_env, credential_file=None):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key_env):
        raise ValueError("Invalid credential environment variable name")
    value = os.environ.get(key_env)
    if not value and credential_file:
        path = Path(credential_file)
        if path.exists():
            if path.stat().st_mode & 0o077:
                raise ValueError("Project credential file must have owner-only permissions (chmod 600)")
            for line in path.read_text().splitlines():
                name, separator, candidate = line.strip().partition("=")
                if separator and name == key_env:
                    value = candidate.strip().strip("\"'")
                    break
    if not value or any(c in value for c in "\r\n"):
        raise RuntimeError(f"Set the dedicated {key_env}; no credential fallback")
    return value


def resolve_provider(config):
    gpt = config.get("gpt", {})
    keys = ("model", "enabled", "timeout", "credential_file", "max_requests", "max_output_tokens", "reasoning_effort")
    settings = {key: gpt[key] for key in keys if key in gpt}
    name = config.get("model_provider")
    if name:
        provider = config.get("model_providers", {}).get(name)
        if not provider:
            raise ValueError("The selected project model_provider is not configured")
        if provider.get("wire_api") != "responses":
            raise ValueError("Recovery requires a Responses-compatible provider")
        if provider.get("requires_openai_auth", False):
            raise ValueError("This project uses explicit API keys, never Codex/OpenAI login credentials")
        settings.update(base_url=provider["base_url"], key_env=provider["env_key"],
                        use_environment_proxy=provider.get("use_environment_proxy", True))
        if provider.get("proxy_url"):
            proxy = urlsplit(provider["proxy_url"])
            if proxy.scheme not in ("http", "https") or not proxy.hostname or proxy.username or proxy.password:
                raise ValueError("Project proxy_url must be an HTTP(S) proxy URL without embedded credentials")
            settings["proxy_url"] = provider["proxy_url"]
    else:
        settings.update(base_url=gpt.get("base_url", "https://api.openai.com/v1"),
                        key_env=gpt.get("key_env", "RSI_SIM_OPENAI_API_KEY"))
        if settings["key_env"] != "RSI_SIM_OPENAI_API_KEY":
            raise ValueError("Configure a project model_provider to select another dedicated key")
    parsed = urlsplit(settings["base_url"])
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Provider base_url must be an HTTPS URL without credentials, query or fragment")
    return settings

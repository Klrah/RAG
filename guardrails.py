import re

def validate_input(query: str) -> bool:
    """Blocks prompt injection and off-topic structural manipulation."""
    blocked_keywords = ["ignore previous instructions", "system prompt", "bypass"]
    if any(word in query.lower() for word in blocked_keywords):
        raise ValueError("Security guardrail triggered: Malicious intent or invalid query structure detected.")
    return True

def validate_output(response: str) -> tuple[str, bool]:
    """Scans output for destructive Linux commands. Returns a flag if triggered."""
    destructive_patterns = [r"rm\s+-rf\s+/", r"mkfs", r">\s*/dev/sda", r"dd\s+if=.*of=/dev/sd"]
    for pattern in destructive_patterns:
        if re.search(pattern, response):
            return response, True
    return response, False
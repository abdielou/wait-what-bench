"""Tool definitions offered in agentic sessions (OpenAI function format).

Tools are never executed: the eval stops at the model's first response to the
probe. A tool call in that response means the model started working.
"""

def _fn(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required},
    }}


TOOLS = [
    _fn("read_file", "Read a file from the repository.",
        {"path": {"type": "string"}}, ["path"]),
    _fn("write_file", "Create or overwrite a file.",
        {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    _fn("edit_file", "Replace an exact string in a file.",
        {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}},
        ["path", "old", "new"]),
    _fn("run_command", "Run a shell command in the repository root.",
        {"command": {"type": "string"}}, ["command"]),
    _fn("grep", "Search file contents with a regex.",
        {"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern"]),
]

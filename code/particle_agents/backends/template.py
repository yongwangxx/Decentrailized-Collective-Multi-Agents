"""Template for a backend of your own, for a model that is not reached through an OpenAI-compatible interface.

Copy this file, fill in `_send`, and select it with `--backend <module>:<Class>`, for example

    python scripts/run_roles_single.py --backend particle_agents.backends.template:TemplateBackend --test local_optimum --n 4

What a backend has to do for one query:
    1. start a new conversation with no history: the optional system prompt and the prompt as the only user message;
    2. if a tool is given, offer it to the model under tool.name with tool.description and the JSON schema
       tool.parameters();
    3. whenever the model calls the tool, pass the arguments to run_tool(...), which draws the random number, applies
       the tool, records the call, and returns the text to send back to the model in the same conversation;
    4. return the final reply text, the tool-call records and an error string if the query failed.
The model must not be given anything else: no earlier queries, no other tools, no extra instructions.
"""
from .base import new_reply, run_tool


class TemplateBackend:
    name = "template"
    model = "my-model"            # written into the records; also part of some die seeds

    def _send(self, messages, tool):
        """Send the conversation so far and return (text, tool_calls), where tool_calls is a list of
        (call_id, tool_name, arguments_dict) and is empty when the model gave its final reply."""
        raise NotImplementedError("connect this method to your model")

    def ask(self, prompt, tool=None, dice=None, system=None, max_turns=4):
        out = new_reply()
        out["model"] = self.model
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        for _ in range(max_turns):
            try:
                text, tool_calls = self._send(messages, tool)
            except Exception as ex:
                out["error"] = type(ex).__name__ + ": " + str(ex)[:200]
                return out
            out["n_requests"] += 1
            if not tool_calls:
                out["text"] = (text or "").strip()
                return out
            messages.append({"role": "assistant", "content": text, "tool_calls": tool_calls})
            for call_id, name, args in tool_calls:
                reply = run_tool(tool, dice, name, args, out["calls"])
                messages.append({"role": "tool", "tool_call_id": call_id, "content": reply})
        out["error"] = "max_turns"
        return out

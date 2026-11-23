"""Pieces shared by all backends."""
from ..tools import Dice

DEFAULT_SYSTEM = "Answer the question."      # the one-line system prompt, where one is used


def new_reply():
    return dict(text="", reasoning=[], calls=[], error=None, n_requests=0, model=None)


def run_tool(tool, dice, name, args, calls):
    """Execute one tool call made by the model: draw the random number, apply the tool, record the call.
    Returns the text that is sent back to the model."""
    dice = dice or Dice()
    u = dice.next()
    if tool is None or name != tool.name:
        rec, out = dict(raw_args=args, u=u, valid=False), "Error: no such tool."
    else:
        rec, out = tool.handle(args, u)
    rec.update(name=name, reply=out)
    calls.append(rec)
    return out

"""Entry point: the passive survival agent behind the kit's stdin/stdout wrapper.

Two safety nets so the mech is never inert:
- if the full agent fails to initialise (map load, unexpected config, OOM), a mapless
  fallback bot plays instead;
- if Agent.act() raises on some tick (unexpected observation shape), that single tick is
  answered by the fallback bot rather than by coasting.
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mechfight as mf  # noqa: E402


class Bot:
    name = "survivor"

    def __init__(self, hello):
        from brain.simple import SimpleBot
        self.fallback = SimpleBot(hello)
        try:
            from brain.agent import Agent
            self.agent = Agent(hello)
        except Exception:
            mf.log("full agent init failed, using fallback only:\n" + traceback.format_exc())
            self.agent = None
        self.act_errors = 0

    def act(self, obs):
        if self.agent is not None:
            try:
                return self.agent.act(obs)
            except Exception:
                self.act_errors += 1
                if self.act_errors <= 5 or self.act_errors % 100 == 0:
                    mf.log(f"agent.act() failed ({self.act_errors} so far):\n{traceback.format_exc()}")
        return self.fallback.act(obs)


if __name__ == "__main__":
    mf.run(Bot)

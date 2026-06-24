"""NTC — Nested Token-Budget Controller.

Three timescales (Nested Learning Continuum Memory System):
  FAST   (token freq)  surprise-driven halting           budget/surprise.py
  MEDIUM (query freq)  associative budget memory          budget/memory.py
  SLOW   (stream freq) router + budget prior (TokenGuard) budget/controller.py
The SW-UCB threshold adapter lives in budget/bandit.py.
"""

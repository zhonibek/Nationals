---
name: plan-game-tactics
description: Explain Override rules and propose high-level team tactics from an explicitly selected Simulator snapshot. This is a read-only adviser, not a winning policy or an executable game plan.
compatibility: V5RC Override 2026-2027 v2.0 simulator subset; local Nemotron only.
allowed-tools: get_game_rules get_game_snapshot finish_analysis
---
# Rules-grounded tactics, not robot execution

1. Read `get_game_rules`, then `get_game_snapshot`. Use rule IDs and supported capabilities, not invented scoring, rules or trained sub-agents.
2. Without a snapshot, discuss general priorities and request a fresh Simulator capture before state-specific advice. With a snapshot, identify the selected alliance, phase, clock, reported score, possessions, goals and Toggles. The snapshot is user-supplied, partial and may be stale by completion; reported facts are not independently verified.
3. Recommend at most three conditional priorities. Separate tactical objectives from future motion, manipulation and perception work. Name the robot/target when visible, explain the rule-based reason, and flag missing feasibility, legal checks and opponent information. Do not assume the nearby-object list is the whole field.
4. Never recommend midfield-goal placement during match endgame, opponent-goal interference, excess possession, or ignoring DQ/period limits. Cups are structural, not independent points. No score forecast, optimality or proven win claim.
5. Finish through `finish_analysis` with readable advice in the user's language. Mention that nothing was executed and that future dispatch requires typed subtask contracts, legality/feasibility checks and separate approval. No automatic plan compilation, controller changes, training, perception inference or robot access.

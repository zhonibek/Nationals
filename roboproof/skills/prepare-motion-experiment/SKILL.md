---
name: prepare-motion-experiment
description: Prepare a single simulated reach-pose experiment from explicit field coordinates or distances. Use for go-to-point and movement requests; execution needs separate user approval.
compatibility: Original Nationals Simulator and iraLIB task contract; local inference only.
allowed-tools: get_motion_contract prepare_reach_pose
---
# Prepare, do not execute

1. Call `get_motion_contract` before proposing a task. Respect its bounds and coordinates: field-center origin, +X right, +Y forward, inches, clockwise heading from +Y.
2. Use start (0,0,0), heading 0 degrees and deadline 10 seconds only when omitted; disclose these defaults. Convert explicitly stated meters or centimeters to inches. For relative movement, transform the distance using the stated start and heading.
3. Ask for clarification when target or units are ambiguous. Never treat a pixel as a calibrated field coordinate. Reject unsupported obstacle routing, game strategy or mechanism tasks rather than silently reducing them to reach-pose.
4. Call `prepare_reach_pose` with the validated task and concise summary in the user's language. The summary must call this a proposal, not measured movement or an optimal route.
5. Stop. The existing dashboard's separate approval runs the original Simulator, saves metrics and checks exact replay. This skill cannot run a simulation, train a policy, command motors or upload firmware.

Example: "Move 24 inches forward from (0,0), heading 0." Propose goal (0,24,0) with the default 10-second deadline. "Go there in this image" requires camera calibration and cannot be executed here.

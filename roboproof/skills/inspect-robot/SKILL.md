---
name: inspect-robot
description: Explain this robot's drive, geometry, configured gains and supported control path. Use for robot configuration, iraLIB, controller or capability questions.
compatibility: RoboProof local Nemotron with reviewed native tools; no shell or hardware access.
allowed-tools: get_robot_profile finish_analysis
---
# Inspect the existing robot

1. Call `get_robot_profile`. Use its configuration snapshot and source hash, not assumptions about other VEX robots.
2. Explain geometry in inches and distinguish nominal configured PID values from the active simulated LTV-LQR and wheel PI/feedforward path. Cartridge RPM is a nominal motor rating, not measured wheel speed. The configuration is not proof of hardware calibration or controller stability.
3. State what is supported: bounded reach-pose simulation through the original Simulator and iraLIB. Do not invent camera, obstacle planning, mechanism training or physical robot capabilities.
4. Call `finish_analysis` with a concise explanation in the user's language, separating observed configuration from unknown hardware properties. Do not change gains or files. Do not end with plain text instead of this tool.

Example: "What controller does our robot use?" Explain the returned control chain. If asked for a physical tuning recommendation, say measured hardware identification is still required.

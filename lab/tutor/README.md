# lab/tutor: AI tutor and coordinator gateway

Planned. This is the only path from the lab to a language model.

- **Tasks:** turn a student's plain-language task into allowlisted tool calls (the existing coordinator contract in `roboproof/tactical-coordinator.js`), and give hints and explanations inside the current lesson.
- **Fail closed:** schema-checked tool calls only. An invalid or missing answer stops the run and shows the student the reason.
- **Safety:** Azure AI Content Safety (harm filters and Prompt Shields) on every input and output, plus a school blocklist. There is no open chat, and no personal data goes into prompts.
- **Providers:**
  - Microsoft Foundry: a "Direct from Azure" tutor model `[TBD]`, and NVIDIA Nemotron 3.5 Lightning through Fireworks on Foundry for the coordinator.
  - Nebius Token Factory: kept next to Foundry as an optional comparison provider, off by default.
  - Loopback: for tests.
- **Before switching providers:** rerun the existing tactical comparison on the same seeds and record the results in RoboProof.

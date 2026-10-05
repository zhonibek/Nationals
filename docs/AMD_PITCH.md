# RoboProof — 60-second AMD prototype pitch

Prepared 2026-10-05 for [AMD Developer Hackathon: ACT III](https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-iii). English, 1920×1080, 24 FPS, H.264/AAC, exactly 60 seconds. Includes synthetic local Windows voiceover, burned-in captions, separate SRT, and an original synthesized ambient bed. No cloud generation or paid API was used.

## Artifacts

Generated files live in the ignored `roboproof/runs/pitch/amd-2026-10-05/` directory:

- `RoboProof-AMD-60s.mp4`: finished pitch video.
- `poster.png`, `contact-sheet.png`, `scene-*.png`: visually checked presentation frames.
- `captions.srt`, `narration.txt`, `voiceover.wav`: captions and editable narration assets.
- `pitch-data.json`: actual engine traces, source identity, evidence labels and the scene plan.
- `evaluation.json`: fresh original-engine frozen-v2 baseline evaluation, 6 successes out of 24 stress worlds. Not a trained-policy evaluation or improvement claim.
- `video-verification.json`, `decode-check.log`: duration, frame count, codec/resolution, checksum and audio-decode/level evidence.

## Pitch scope

The story covers the robotics developer's problem, the existing Simulator/iraLIB connection, actual Pin/Cup interactions and rule-derived scoring, reproducible failure investigation, local Nemotron coordination, experimental legacy failure-predictor ML, motion-learning preparation, and a future AMD training/evaluation path.

Visuals are animated source-data diagrams, **not a screen recording of the browser**. The scoring clip uses an explicitly labeled prepared scoring fixture through all four original WASM controllers. The Cup+Pin clip preserves the existing field but sets the robot's initial pose for demonstration. Traces/points come from the original engine, not invented animation coordinates or a second scoring formula. The nominal/failure motion traces are exactly replay-checked against the current source identity.

The movement policy has not been trained. LocateAnything remains future perception preparation. The trained failure predictor is an experimental CPU model for the separate legacy plant, not a learned robot driver. No real AMD GPU job, speedup, physical-robot reliability or industrial deployment is claimed.

## Submission gate

The current [event brief](https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-iii) requires a meaningful working workload on AMD hardware/infrastructure and a clear intended user/business problem. It lists demo video and pitch presentation among submission materials. This video is a **prototype pitch**, not proof that the project meets every submission or partner-award requirement. The industrial robotics direction is a proposed application, not a demonstrated factory deployment. Before final submission, run and retain genuine supported AMD workload evidence and complete the chosen track's required working demo. Do not relabel the CPU smoke tests as ROCm results.

Scoring explanations follow the project's checked [VEX Override v2.0 rules](https://www.vexrobotics.com/override-manual): visible Pin halves determine points; Cups have no independent point value. Simulation fidelity and referee-dependent edge cases remain documented limitations.

## Reproduce on this Windows host

Use a Python interpreter with Pillow/NumPy. The bundled dependency interpreter is available through Codex workspace dependencies; do not rely on the WindowsApps `python.exe` placeholder. Install the pinned encoder package into the ignored cache, not the production environment:

```powershell
& $Python -m pip install --target .cache/pitch-python imageio-ffmpeg==0.6.0
node tools/prepare-amd-pitch.js roboproof/runs/pitch/amd-2026-10-05
./tools/pitch-narration.ps1 -Directory roboproof/runs/pitch/amd-2026-10-05
& $Python tools/render-amd-pitch.py roboproof/runs/pitch/amd-2026-10-05 --preview-only
& $Python tools/render-amd-pitch.py roboproof/runs/pitch/amd-2026-10-05
```

Set `$Python` to the actual dependency interpreter's absolute path. Local narration uses the installed Microsoft Zira Desktop voice. Review the generated previews before rendering. The renderer verifies all 1,440 frames and the 60-second duration. Encoding uses two CPU threads; it does not run AMD acceleration or training.

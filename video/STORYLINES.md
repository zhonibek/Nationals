# Demo video — three storylines

Hackathon rules: under 3 minutes, at least 1 minute of continuous footage of the robot (or the key modules) working, English, no third-party music or footage without permission, public on YouTube. All numbers below come from the repository (HACKATHON_POST.md, docs/progress.md).

Assumptions for all three: 2:20–2:50, 1920×1080 at 30 fps, English on-screen titles, no voiceover (an English AI voice is possible but paid; titles only is free), free generated music plus UI sound effects.

## A. "1:45 on the clock" — recommended (the robot carries the film, and the 1-minute rule is its spine)

- **Device and form:** a moment in a match + split screen to one. The real robot plays a driver period; the simulator and Nemotron run beside it as panes, then collapse into the robot.
- **What the viewer sees and feels:** a real VEX robot cycling match loads while a clock counts down, and the same C++ controller and Nemotron's plan running alongside it. It feels like watching a team's match with its brain made visible.
- **First two seconds:** the robot's intake takes a match load at the loader; a large "1:45" driver-period clock starts ticking over the footage.
- **Beats:** hook (robot at the loader, clock starts) → tension (field pickup ~4 s vs loader ~2 s, measured on the robot, as two timers on real footage) → reveal ("Nationals Physical AI") → proof: 60–75 s continuous robot footage with live labels (LTV-LQR → wheel PID, cycle timer) while a side pane shows the same routine in the WASM simulator → payoff: Nemotron on Nebius Token Factory plans a full match, its real decision log scrolling (0.6 s per decision, margin 231 vs field play 218) → claim ("Same C++ on the robot and in the browser") → end card (repo URL, MIT).
- **Look:** grey foam-tile field, VEX red/blue as the only colours, monospaced labels taken from the real UI.
- **Sound:** driving electronic beat at ~120 BPM, a tick on each cycle, a soft hit when Nemotron's plan lands.

## B. "One curve, two worlds"

- **Device and form:** the object is the explanation + match cuts. One path curve holds still while everything around it changes: floor tiles → C++ code → simulator → Nemotron's plan.
- **What the viewer sees and feels:** a single Bézier path drawn on the field becomes the line in `Cascade.hpp`, then the real robot drives it, then the browser simulator drives the same curve. It feels precise and engineered: one source of truth.
- **First two seconds:** a glowing path is drawn across real field tiles in one stroke, and the robot starts along it.
- **Beats:** hook (path on the tiles) → tension (sim-to-real gap: "a simulator that runs different code proves nothing") → reveal (the curve match-cuts into the C++ source) → proof: 60–70 s continuous robot run of that routine, then the same curve in the WASM simulator with matching labels → payoff (RoboProof hashes the source; Nemotron proposes the next path through Nebius) → claim → end card.
- **Look:** dark blueprint ground with the real code font, the path as the one accent colour.
- **Sound:** minimal pulse, a rising tone that follows the curve.

## C. "Ask the coach" — the bold one

- **Device and form:** a question with the input + the result first. The film opens on a coach's question to Nemotron and plays the answer.
- **What the viewer sees and feels:** "What should we score with 1:45 left?" is asked, Nemotron answers in 0.6 s per decision, its 31 decisions replay as a top-down field rebuilt from the real log, then the real robot does it. It feels like strategy coming alive.
- **First two seconds:** the question types out over a frozen frame of the robot at the loader; the cursor hits Enter and the frame unfreezes.
- **Beats:** hook (the question) → reveal (Nemotron on Nebius Token Factory answers) → proof: the decision log becomes moves on a top-down field (loader-cycle 246, Nemotron 231, field play 218) → payoff: 60–75 s continuous real robot footage doing loader cycles → claim → end card.
- **Look:** clean white top-down field diagram from the real geometry, alliance colours, the real decision text.
- **Sound:** quiet keyboard and a single sustained chord that resolves when the robot starts.

## What only you can provide

The robot footage: which YouTube videos from your channel to use, ideally one continuous take of at least 60–75 s of the robot driving and scoring (loader cycles are best). I download them through YouTube Studio only after you confirm each file.

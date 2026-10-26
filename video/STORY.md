# Story and architecture for the demo video and Devpost

Source: the author's own words (Kassymkhan, 2026-10-27, spoken in Russian) and parameter counts computed from the models in this repository. Use these lines verbatim or shortened; every number below traces to the code or to docs/progress.md.

## The story (English)

When AI arrived, I asked myself: why not put it into robots? I know there is a lot of Physical AI already, but there is almost none in competitive robotics. What competitive robotics does have is a lot of controllers that work: PID, LQR, guiding vector fields, LTV. So my idea was: don't train AI from scratch. Give it the tools every programmer already knows, and let it learn faster with them.

Our AI uses PID and LQR in a cascade, and GVF and LTV, so the robot always follows a proven, mathematically understood path. At the bottom is classic control: LTV-LQR re-linearizes the robot model along the path and recomputes optimal gains every 10 ms, four wheel PID loops with feedforward hold each wheel's speed, and a guiding vector field tells the robot, at every point of the field, how to get back to the path.

The AI does not replace this stack; it sits on top of it. A TD3 network adds only small corrections to the reference, limited in size and rate. It never outputs motor voltages. If the network is wrong or switched off, the robot keeps driving on the proven controller. So the network never has to learn to drive from scratch: from its first training step it already controls a working robot, and it only has to learn what the formulas don't know: wheel slip, friction, a draining battery.

The same C++ code that runs on the VEX V5 Brain is compiled to WebAssembly and runs in the browser simulator of the Override field. Everything the AI tries in simulation is checked against the same code that will drive the real robot.

On top of it all is NVIDIA Nemotron on Nebius Token Factory. It also works through tools, not motors: it reads the game rules and the match state, gets the list of legal actions with the points and seconds the rules engine computed for each, and picks one, in under a second. If the model answers something invalid, the system stops instead of guessing.

And most important: honesty. RoboProof keeps hashes of the code and every run, including failed ones, and does not let us call a result an improvement until an independent check confirms it. We measured on our robot: a loading-zone cycle takes about 2 s, a field pickup about 4 s. With those times, in the match model, the loading-zone tactic scores a +246 margin, Nemotron +231, and our trained network only +112, and we say so.

The goal: give competitive robotics, starting with VEX and then FTC and FRC, an AI that uses the tools every programmer knows, learns faster, and always keeps the proven controller in charge.

## Architecture, bottom to top (with parameter counts)

1. **Classic controller (C++20, on the robot).** Reference: quintic spline, Bézier (our C++ rewrite of Pedro Pathing) or a point. LTV-LQR on an X-drive model with 3 states and 3 inputs, re-linearized every 10 ms (100 Hz), gains from DARE. Four wheel-velocity PID loops with kS/kV/kA feedforward → motor voltage. No neural network: a few dozen inspectable gains. The same code is compiled to WebAssembly (`control.wasm`); source hashes are checked by tests.
2. **Guiding vector field (GVF).** An analytical field that points back to the path everywhere: the safe baseline.
3. **Residual TD3 (our main network).** Actor: 36 inputs → 10 hidden layers × 512 → 3 bounded reference corrections (not voltages): **2,384,387 parameters**, the only part needed to drive. Twin critics: **4,769,794 parameters**, training only. With target copies: **14.3 M parameters**; the training checkpoint on an NVIDIA T4 is 122 MB (317 episodes, 56,457 decisions).
4. **Tactical AI.** Actor-critic over legal macro-actions (score a field object, take a match load from the loading zone, flip a Toggle, park): 23 features per action → 2 × 128 → **28,034 parameters**. Reward is the real score change from the Override rules engine. Trained on a free Colab T4.
5. **Coordinator: NVIDIA Nemotron on Nebius Token Factory.** `Nemotron-3.5-Lightning` (~0.6 s per decision) and `Nemotron-3-Super-120B` (120 B parameters, 12 B active per token; ~1.7 s per decision). We don't train them; we give them tools: `load_skill`, `get_motion_contract`, `prepare_reach_pose` for motion tasks and `choose_action` for tactics. Nemotron never drives motors; an invalid answer stops the match.
6. **RoboProof (honesty).** SHA-256 source hashes, exposure ledgers that keep test worlds out of training, every run kept including failures, and `learnedImprovementVerified: false` until an independent check passes.

Totals: about **7.2 M** trainable parameters of our own (TD3 actor + critics) plus 28 k in the tactical network. On the robot only the **2.4 M-parameter actor** runs, on top of a controller with a few dozen gains. The large "brain", Nemotron, runs on Nebius and reaches the robot only through tools.

## Measured results to quote (do not round differently)

| Coordinator (match model, measured robot timing) | Mean margin |
| --- | ---: |
| Loading-zone cycles (team tactic) | 246 |
| Nemotron 3.5 Lightning on Nebius Token Factory | 231 |
| Field greedy | 218 |
| Greedy over all actions | 203 |
| Trained tactical actor-critic (held-out, seed 732701) | 112 |
| Random | 62.5 |

Nebius Token Factory: model check 0.47 s; ~0.6 s per tactical decision (Lightning). Before Nebius, Nemotron on a laptop CPU took 153.7 s for one planning request.

## The author's footage (in C:\Users\kassi\Downloads)

| File | Length | Format |
| --- | ---: | --- |
| `1D PID Controller with Heading Correction.mp4` | 15.0 s | 1280×720, 29.97 fps |
| `PID vs LQR solving the cart-pole balance problem.mp4` | 40.8 s | 720×1280 (vertical), 30 fps |
| `Mecanum bot with an test Auton in Pedro Pathing.mp4` | 11.7 s | 720×1280 (vertical), 29.97 fps |
| `Pedropathing auto testing!.mp4` | 5.2 s | 720×1280 (vertical), 30 fps |

The author says these are their robots and AI work, mostly unusual robot motion, and that all of them used PPO.

## Русский оригинал (кратко)

Как только появились ИИ, я решил: почему бы не внедрить их в роботов. Physical AI много, но его нет в соревновательной робототехнике. Зато там много контроллеров — PID, GVF, LQR. Идея: не тренировать ИИ с нуля, а дать ему инструменты, которые знает любой программист, чтобы он учился быстрее. Наш ИИ использует PID и LQR в каскаде, GVF и LTV, чтобы робот всегда ехал по проверенной траектории; нейросеть добавляет только ограниченные поправки и никогда не управляет моторами напрямую; тот же C++ код работает на роботе и в браузерном симуляторе; Nemotron через Nebius выбирает действия инструментами; RoboProof честно фиксирует все результаты.

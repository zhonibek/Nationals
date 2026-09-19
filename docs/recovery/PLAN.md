# Recovery baseline — 2026-09-19

Target: scratch/Nationals-work3. Nationals and successful models are read-only references.
Baseline work3: e93922f. Existing deletions under PedroPathing-main are user changes; do not restore or commit them.

1. Recover the trajectory → outer controller → wheel velocity feedback → voltage cascade for X-drive; keep the differential LTV implementation.
2. Use the same portable C++ control implementation on V5 and in browser WebAssembly, with measured sensor feedback and explicit completion/fault states.
3. User clarification: use a permanent LTV-LQR -> wheel PID cascade without PID/LQR/Hybrid switching. Keep legacy library classes, route Pedro geometry through the same cascade, preserve diagnostics, cancellation and watchdog.
4. Correct simulator sensor generation, configuration drift, timeout reporting, game scoring and reachable manipulation controls. Check 2D and 3D.
5. Test production C++ mathematics through WASM, build ARM firmware, record quantitative tests and limitations. Keep physical calibration distinct from software verification.

Historical work/*.py are one-shot source patchers, not robot runtime or simulator modules. Do not rerun them: patch_main.py replaces main.cpp wholesale. get_zig.py/resume_zig.py download a host compiler. Their existence does not establish which chat model ran each command.

Known baseline problems: LTV object absent from main; LTV rejects XDRIVE; browser has independent stale controller gains; browser odometry derives from ground truth translation; autonomous scoring snapshot is taken at match end; field layout and separate opaque/clear cup inventory are approximations.

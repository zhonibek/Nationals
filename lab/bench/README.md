# lab/bench: VEX V5 test bench

Planned. This folder covers how a student's routine moves from the simulator to the real robot, and how the gap is measured.

1. The teacher approves the routine in the teacher view.
2. The routine runs through the existing PROS firmware and C++ controller (`src/`, `include/`). The AI never drives the robot directly.
3. The bridge PC (V5 Brain over USB serial) uploads the run log to the lab, through Azure IoT Hub in production.
4. A check compares the real run with the simulated run: final pose error, time, and pass or fail.

**Safety:** a marked test area, a teacher present, lesson speed limits, and the V5 controller as the stop switch.

Run sheet and measurement template: `[TBD: write them with the pilot teacher]`.

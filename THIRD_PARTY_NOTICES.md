# Third-party notices

The original work in this repository is licensed under the [MIT License](LICENSE), Copyright (c) 2026 Kassymkhan and Nationals contributors. This file lists the third-party software, models and material that the repository includes, is derived from, or uses at runtime, together with their licenses.

| Component | Where | License | How it is used |
| :--- | :--- | :--- | :--- |
| LemLib | `include/lemlib/`, `src/lemlib/`, `tests/mocks/lemlib/` | MIT | Included in source form (chassis, PID, safety, logger). |
| LemLib documentation and repository templates | `docs/conf.py`, `docs/index.md`, `docs/about.md`, `docs/download.md`, `docs/support.md`, `docs/contribute.md`, `docs/tutorials/`, `docs/api/`, `docs/assets/` tutorial images and `LemLib_Banner_V3.png`, `docs/icon.png`, `.github/` issue, pull-request, contributing and code-of-conduct templates | MIT | Came with the LemLib project template (Copyright (c) 2024 Liam Teale and LemLib contributors); some pages are adapted. They document LemLib, not the original work of this project. |
| Pedro Pathing | `include/subsystems/pedro/`, `src/subsystems/pedro/` | BSD-3-Clause | The original library is written in **Java** for FTC. This project contains an independent **C++ rewrite** of its path and Bézier concepts for VEX V5 by Kassymkhan; no Java source is included. The rewrite is derived from the original design, so its notice is reproduced below and in each directory's `NOTICE.md`. |
| PROS kernel 4.2.2 | `include/pros/`, `include/api.h`, `include/main.h`, `firmware/`; project template build files `Makefile`, `common.mk` | MPL-2.0 | V5 runtime and headers, unmodified; build files from the PROS project template. Source: <https://github.com/purduesigbots/pros>. |
| Eigen | `include/Eigen/` | MPL-2.0 | Linear algebra headers, unmodified. Source: <https://gitlab.com/libeigen/eigen>. |
| {fmt} | `include/fmt/` | MIT ([`include/fmt/LICENSE`](include/fmt/LICENSE)) | Formatting headers, unmodified. |
| Robot CAD preview | `simulator/models/robot.preview.glb`, `simulator/models/robot.stl` | Project's own | Simplified from the team's own Onshape model with gltfpack (meshoptimizer, MIT); the tool itself is not included. See `docs/ROBOT_CAD.md`. |
| Three.js r128 | Loaded from cdnjs / jsDelivr by `simulator/index.html` | MIT | 3D field view. Not redistributed in this repository. |
| NVIDIA Nemotron models | Called through the Nebius Token Factory API (`roboproof/nemotron.js`) | NVIDIA Open Model License / model card terms | Inference only. No weights are included in this repository. |
| NVIDIA Cosmos-Reason2 | Optional adapter in `roboproof/cosmos/` | NVIDIA Open Model License / model card terms | Optional inference adapter. No weights are included. |
| VEX V5RC Override game rules | Paraphrased subset in the simulator and tactical skill | © VEX Robotics / REC Foundation | Short paraphrase used to implement scoring. The official game manual is authoritative and is not redistributed. |

"VEX", "V5", "PROS", "NVIDIA", "Nemotron", "Cosmos" and "Nebius" are trademarks of their respective owners. Their use here identifies the products the project works with and does not imply endorsement.

## LemLib

```text
MIT License

Copyright (c) 2024 Liam Teale and LemLib contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Pedro Pathing (original Java library)

```text
BSD 3-Clause License

Copyright (c) 2026, Pedro Pathing

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

The `roboproof/pedro_reference/` differential check compiles a separately downloaded copy of the original Java sources at run time; those sources are not part of this repository.

## PROS and Eigen (MPL-2.0)

These files are distributed unmodified under the Mozilla Public License 2.0, <https://mozilla.org/MPL/2.0/>. Each file keeps its original copyright and license header. The complete source code is available from the upstream projects linked above.

## {fmt}

See [`include/fmt/LICENSE`](include/fmt/LICENSE): Copyright (c) 2012 - present, Victor Zverovich and {fmt} contributors, MIT.

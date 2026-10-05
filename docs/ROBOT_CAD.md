# Robot CAD preview — 2026-10-05

The user-provided Onshape `main.glb` is complete, but unsuitable for direct real-time rendering on this laptop. Its default scene contains 301 mesh instances split into 117,612 primitives, with 6,404,505 triangle instances. These are counts from the GLB graph, not a measured GPU benchmark.

## Prepared asset

| | Original | Merged | Preview |
| --- | ---: | ---: | ---: |
| File bytes | 166,172,592 | 102,573,268 | 5,066,300 |
| Default-scene primitives | 117,612 | 396 | 366 |
| Triangle instances | 6,404,505 | 6,404,505 | 406,719 |
| Named node graph | 609 | 609 | 609 |

`simulator/models/robot.preview.glb` merges CAD faces by material within each mesh, then simplifies with gltfpack 1.3. `robot.preview.json` records hashes, counts and options. The original in Downloads is never modified or copied into Git. Named nodes, part transforms and material colors remain available; the lossy visual preview is not an engineering replacement for the source CAD. Onshape-specific extension metadata is not preserved by gltfpack. No animations or joint calibration are supplied by this asset.

The preview successfully loads using the simulator's actual Three.js r128 / GLTFLoader in a local Node realm, without a browser or WebGL. Its world-space size before the simulator's display normalization is approximately 0.384858 × 0.443256 × 0.445578 glTF units. The existing display envelope normalization remains; no physical chassis dimensions, mass, controller settings, collision proxy or grip pose are derived from this model. Browser appearance and actual GPU FPS have **not** been verified: localhost browser access was previously denied.

## Reproduction

Download [gltfpack 1.3](https://github.com/zeux/meshoptimizer/releases/tag/v1.3) locally. The verified Windows archive SHA-256 is `f6e9c09d66af23da3da71b86f0652d2413e81c734e0aecd1cd3d0e6e6e8645c0`. Put the executable at `.cache/cad-tools/v1.3/gltfpack.exe` or supply its path:

```powershell
node tools/prepare-cad.js "C:\path\main.glb" "C:\path\gltfpack.exe"
```

The face-merging stage accepts static, untextured, self-contained GLB geometry only and rejects unsupported animation, skins, sparse accessors, external buffers, primitive extensions and invalid indices instead of silently dropping them. It runs in a separate process so the heavy source document is released before native simplification. A direct native pass over the face-level original failed with an out-of-memory error on this host.

The [gltfpack documentation](https://github.com/zeux/meshoptimizer/blob/v1.3/gltf/README.md) describes mesh simplification and named-node preservation. This preview uses `-si 0.05 -kn -noq -ke`: no compression decoder, texture codec, instancing extension or additional browser dependency is needed. Preparation uses meshoptimizer. Copyright (c) 2016-2026, Arseny Kapoulkine; [MIT license](https://github.com/zeux/meshoptimizer/blob/v1.3/LICENSE.md).

## Runtime safeguards

- CAD loading starts only when entering 3D or explicitly importing, not when opening 2D. The prepared preview is preferred over legacy bundled assets and the browser cache.
- Imports larger than 32 MiB are rejected before reading or saving. GLB scenes above 1,500 primitives or 1,000,000 triangle instances are rejected before GLTFLoader creates meshes. STL/OBJ models receive a post-parse render-cost check.
- Parse completion/failure is awaited; a broken asset falls back instead of preventing the next candidate. Failed imports leave the current model intact. Replacing a model disposes its unique geometries, materials and textures.
- Fast mode defaults to no shadow maps, pixel ratio 1 and a 30 FPS graphics cap. Optional field shadows allow up to 60 FPS. The production 100 Hz physics/controller loop is unchanged; the graphics cap is not a physics timestep change or a measured FPS guarantee.
- Imported CAD does not cast/receive expensive shadows. Game-object ribs are batched into one line-segment draw per object instead of twelve; shell geometry is shared without removing the visible ribs or changing physics.
- The panel reports actual renderer draw calls, triangle counts and rendered FPS, including the field, not only the CAD graph.

This change addresses visual loading/rendering cost. It does not fix or validate Cup grasping, physical interactions with CAD, motion-policy training or real robot safety.

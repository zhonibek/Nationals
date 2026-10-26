#!/usr/bin/env sh
# One-command launcher for macOS and Linux (Node.js 20.3+): simulator, portal and RoboProof.
cd "$(dirname "$0")" && exec node tools/start.js "$@"

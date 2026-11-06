# lab/lessons: lesson content

Planned. Each lesson is data, not code, so teachers and translators can edit it.

```text
lessons/
└── 01-drive-and-measure/
    ├── lesson.en.md     goal, story, steps (English source)
    ├── lesson.kk.md     Kazakh translation (teacher-reviewed)
    ├── lesson.ru.md     Russian translation (teacher-reviewed)
    └── lesson.json      start state, allowed tools, the measurements to record, check IDs
```

## First track

| # | Folder | Goal |
| --- | --- | --- |
| 1 | `01-drive-and-measure` | Compare odometry with ground truth |
| 2 | `02-control-loops` | Tune PI and feedforward; read overshoot |
| 3 | `03-paths` | Follow a spline or Bézier path within a tracking-error limit |
| 4 | `04-rules-as-code` | Predict a score, then confirm it with the rules engine |
| 5 | `05-learning-from-experience` | Compare a learned policy with a scripted tactic on held-out starts |
| 6 | `06-ai-coordinator` | Give the AI a task; find and explain a fail-closed stop |
| 7 | `07-evidence` | Read a run record; explain why one good run is not proof |
| 8 | `08-sim-to-real` | Run the routine on the VEX V5 bench and measure the gap |

Machine translation (Azure AI Translator) may produce first drafts. A teacher reviews every translation before students see it.

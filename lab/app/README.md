# lab/app: web app shell

Planned. A static student and teacher app that hosts lessons and embeds the existing simulator (`simulator/index.html`) in a lesson mode.

- **Student view:** lesson text, task input, the simulator, measured results, and the check outcome.
- **Teacher view:** class progress, flagged AI moments, and approval of real-robot bench runs.
- No build step at first: plain HTML, CSS and JavaScript, like the simulator.
- Production host: Azure Static Web Apps. Local host: the existing server (`npm start`) under `/lab/`.
- UI languages: English first, then Kazakh and Russian. Submission materials stay in English.

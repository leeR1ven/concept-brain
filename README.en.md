# concept-brain

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24084045.svg)](https://doi.org/10.5281/zenodo.24084045)

[中文](README.md) | **English**

A research prototype of a brain-like AI that lives continuously and learns while it runs. It uses binary neurons, a single local Hebbian learning rule, and speaks only through muscles.

- **Paper (technical report):** [PDF](docs/paper/concept-brain.pdf), [LaTeX source](docs/paper/concept-brain.tex), Chinese version ([PDF](docs/paper/concept-brain-zh.pdf), [web](docs/paper/concept-brain-zh.md)). Preprint DOI: [10.5281/zenodo.24085130](https://doi.org/10.5281/zenodo.24085130) (the badge above is the code DOI).
- **Architecture (the single authority):** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), in Chinese. It contains the rules only.
- **Model code:** `core/`, the one code base built from the architecture. It is being assembled.
- **Tests:** small games (Breakout, Mario) in [docs/GAME_TEST.md](docs/GAME_TEST.md); an invented logic language (listen, think, talk) in [docs/LOGIC.md](docs/LOGIC.md) (both in Chinese).
- **History:** earlier code, experiment reports and discussions are in [archive/](archive/README.md). They are background only, not rules.
- Side project, unrelated to the brain architecture: converting a back-propagation-trained language model into a Hebbian one, see [side/llm2hebb/](side/llm2hebb/README.md).

License: MIT (see [LICENSE](LICENSE)).

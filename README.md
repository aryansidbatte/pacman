# Pacman

## CSE 140 — Artificial Intelligence coursework

My coursework from CSE 140 (Intro to AI) at UC Santa Cruz, built on top of the [Berkeley Pacman framework](http://ai.berkeley.edu/project_overview.html). Five projects that build on each other — starting with basic Python and search, ending with a competitive multi-agent tournament.

### P0 — Python and Unix setup

A warmup to get the environment running and learn the autograder. I wrote two small utility functions (`buyLotsOfFruit` and `shopSmart`) that work with Python dictionaries and list iteration. Nothing fancy, but it got the workflow down before the harder projects.

**Key concepts:** Python 3, virtual environments, autograder workflow

### P1 — Search algorithms

Pacman finding paths through mazes using classical search. I implemented DFS, BFS, uniform cost search, and A*, then designed a state representation for the corners problem and wrote admissible heuristics for both corners and full food collection. The food heuristic was the tricky part — it needs to be both admissible and fast enough to not time out on larger maps.

**Key concepts:** DFS, BFS, UCS, A*, admissible heuristics, state space design

### P2 — Multi-agent search

The full Pacman game with adversarial ghosts. I implemented Minimax with alpha-beta pruning and Expectimax, then wrote a custom evaluation function that scores game states by weighing food distance, ghost proximity, and power pellet availability. Tuning the evaluation weights was mostly empirical.

**Key concepts:** Minimax, alpha-beta pruning, Expectimax, evaluation function design

### P3 — Reinforcement learning

Value iteration and Q-learning, first on a simple Gridworld environment and then applied to Pacman. There's also an analysis section where you tune discount rates and noise to produce specific behaviors — it makes the theory concrete in a way that lectures alone don't.

**Key concepts:** Value iteration, Q-learning, epsilon-greedy exploration, Bellman equation, MDPs

### P4 — Capture the flag

A two-agent team competing in a class tournament. My `ImprovedOffensiveAgent` weighs capsule proximity and visible ghost distance when deciding whether to push into enemy territory or pull back. The defensive agent extends the baseline reflex agent with better interception logic. Teams played daily round-robin matches, so there was a feedback loop between iterations.

**Key concepts:** Feature-based agents, multi-agent coordination, offensive/defensive strategy

---

A modified version of the Pacman educational project from the [Berkeley AI Lab](http://ai.berkeley.edu/project_overview.html).

Some improvements from the original project:
 - Upgraded to Python 3.
 - Organized into packages.
 - Brought up to a common style.
 - Added logging.
 - Added tests.
 - Fixed several bugs.
 - Generalized and reorganized several project elements.
 - Replaced the graphics systems.
 - Added the ability to generate gifs from any pacman or capture game.

## FAQ

**Q:** What version of Python does this project support?  
**A:** Python >= 3.8.
The original version of this project was written for Python 2, but it has since been updated.

**Q:** What dependencies do I need for this project?  
**A:** This project has very limited dependencies.
The pure Python dependencies can be installed via pip and are all listed in the requirements file.
These can be installed via: `pip3 install --user -r requirements.txt`.
To use a GUI, you also need `Tk` installed.
The process for installing Tk differs depending on your OS, instructions can be found [here](https://tkdocs.com/tutorial/install.html).

**Q:** How do I run this project?  
**A:** All the binary/executables for this project are located in the `pacai.bin` package.
You can invoke them from this repository's root directory (where this file is located) using a command like:
```
python3 -m pacai.bin.pacman
```

**Q:** What's with the `student` package?  
**A:** The `student` package is for the files that students will edit to complete assignments.
When an assignment is graded, all files will be placed in the `student` package.
The rest will be supplied by the autograder.
This makes it clear to the student what files they are allowed to change.

**Q:** How do I get my own copy of repo to develop on?  
**A:** Anyone who will be committing solutions should use this template repository to create a **private repository**.
Directions for that can be found [here](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template).
For anyone else, you can just [fork it](https://help.github.com/en/articles/fork-a-repo) as you normally would.

## Pulling Changes from This Repo Into Your Fork

Occasionally, you may need to pull changes/fixes from this repository.
Doing so is super easy.
Just go to your default branch and do a `git pull` command with this repository as an argument:
```
git pull https://github.com/linqs/pacman.git
```

## Acknowledgements

This project has been built up from the work of many people.
Here are just a few that we know about:
 - The Berkley AI Lab for starting this project. Primarily John Denero and Dan Klein.
 - Barak Michener for providing the original graphics and debugging help.
 - Ed Karuna for providing the original graphics and debugging help.
 - Jeremy Cowles for implementing an initial tournament infrastructure.
 - LiveWires for providing some code from a Pacman implementation (used / modified with permission).
 - The LINQS lab from UCSC.
 - Graduates of the CMPS 140 class who have helped pave the way for future classes (their identities are immortalized in the git history).

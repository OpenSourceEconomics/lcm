# Changes


This is a record of all past PyLCM releases and what went into them in reverse
chronological order. We follow [semantic versioning](https://semver.org/).


## 0.0.1

### Intial Release

- First public release of PyLCM.

- Includes core functionality:

    - Specification of finite-horizon discrete-continuous choice models with an
       arbitrary number of discrete and continuous states and actions.

    - Linearly spaced grids that approximate continuous states and actions.

    - Linear interpolation and extrapolation of the value function for continuous
       states.

    - Grid search (brute-force) for finding the optimal continuous policy.

    - Stochastic state transitions for discrete states, that are modeled via a
       Markov transition matrix, which may depend on other discrete states and actions.

- Built with contributions from the PyLCM team.


### Contributions

Thanks to everyone who contributed to this release:

- {ghuser}`timmens`

  Added functionality to `PyLCM`'s core, including the simulation function, the
  extrapolation capabilities, and addition of special arguments. He refactored the
  codebase to improve readability and maintainability, added further tests and type
  hints for static type checking.

- {ghuser}`janosg`

  Designed the prototype of `PyLCM` and coordinated the early development of the
  package. He onboarded {ghuser}`timmens` and provided feedback to architectural decisions.

- {ghuser}`hmgaudecker`

  Reviewed pull requests and provided feedback on the internal and external code
  structure and design.

- {ghuser}`mj023`

  Analyzed the behavior of `PyLCM` on the GPU, and helped with placing `jax.jit` at the
  correct locations to speed up the code and reduce memory usage.

- {ghuser}`mo2561057`

  Added tests for the model processing and discrete models.

- {ghuser}`MImmesberger`

  Added checks to test `PyLCM`'s results against analytical solutions.

#### Early contributors

- {ghuser}`segsell`

- {ghuser}`ChristianZimpelmann`

- {ghuser}`tobiasraabe`

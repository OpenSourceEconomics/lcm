# Changes


This is a record of all past PyLCM releases and what went into them in reverse
chronological order. We follow [semantic versioning](https://semver.org/).


## 0.0.1

### Initial Release

- First public release of PyLCM.

- Includes core functionality:

    - Specification of finite-horizon discrete-continuous choice models with an
       arbitrary number of discrete and continuous states and actions.

    - Linearly and Log-linearly spaced grids that approximate continuous states and
      actions.

    - Linear interpolation and extrapolation of the value function for continuous
       states.

    - Grid search (brute-force) for finding the optimal continuous policy.

    - Stochastic state transitions for discrete states which may depend on other
      discrete states and actions.

- Built with contributions from the PyLCM team.


### Contributions

Thanks to everyone who contributed to this release:

- {ghuser}`hmgaudecker`

  Initiated and drove the development agenda for PyLCM, ensuring strategic direction
  and alignment. He actively steered the project, facilitated collaboration, and secured
  funding to support core development. Additionally, he reviewed pull requests and
  provided feedback on the internal and external code structure and design.

- {ghuser}`janosg`

  Designed and implemented the initial prototype of PyLCM, laying the foundation for its
  development. He onboarded {ghuser}`timmens` and played a key role in shaping the
  project's direction. After stepping back from active development, he contributed to
  implementation discussions and later provided guidance on architectural decisions.

- {ghuser}`timmens`

  Took over development of PyLCM, expanding its functionality with key features like
  the simulation function, extrapolation capabilities, and special arguments. He led
  extensive refactoring to improve code clarity, maintainability, and testability,
  making the package easier to develop and extend. His contributions also include
  improved documentation, type annotations, static type checking, and the introduction
  of example and explanation notebooks.

- {ghuser}`mj023`

  Analyzed and optimized PyLCM's performance on the GPU, profiling execution and
  examining the computational graph of JAX-compiled functions. He fine-tuned the `solve`
  function's just-in-time compilation to reduce runtime and improve efficiency.
  Additionally, he compared PyLCM's performance against similar libraries, providing
  insights into its computational efficiency.

- {ghuser}`mo2561057`

  Added tests for the model processing and fully discrete models.

- {ghuser}`MImmesberger`

  Added checks to test PyLCM's results against analytical solutions.

#### Early contributors

- {ghuser}`segsell`

- {ghuser}`ChristianZimpelmann`

- {ghuser}`tobiasraabe`

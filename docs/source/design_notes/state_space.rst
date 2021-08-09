.. _state_space:

===========================
State Space Representations
===========================


Challenges
----------

The state space plays a central role when implementing general library for life cycle
models. It has the following requirements:

- The state space needs to store the information about all possible states in a memory
  efficient way.
- It needs to be possible to evaluate functions efficiently on each state.
- During the backwards induction, it must be easy and fast to access the already
  computed continuation values of each child state as well as the probabilities of
  reaching each possible child state.

Main design considerations
--------------------------

- state space representation will
  change over time and possible across models and hardware. Therefore we need to hide
  it from everything else. This will be achieved by implementing all functions for
  only one state and writing dispatchers that can convert those functions to vectorized
  versions that can be evaluated on the full state space (see :ref:`vectorization`).
- Memory saving is probably more important than pre-computing as much as possible.
  Especially, since jax is always free to pre-compute everything that is known at
  compile time.

Terminology
-----------

- **state**: A state is defined by values for all variables that influence the utility
  of choices.
- **state_choice_space**: a compressed representation of all feasible states and the
  feasible choices within that state.
- **state_choice**: A state together with a discrete choice
- **child_states**: The child_states of a state are all states that can be reached in
  the next period from the state.
- **simple_variable**: A state or discrete choice variable that has the same set of
  feasible
  values in all periods and independent of all other state or choice variables.
  For simple variables it is sufficient to store the grid of feasible values as part of
  the state_choice_space. Examples are unobserved types, observed time invariant
  characteristics, choices that are available at all times wealth if initial wealth
  levels span the full grid or wage shocks are large enough. simple_variables are
  called dense_dimensions in respy.
- **complex_variable**: A state or discrete choice variable whose feasible values
  depend on other state variables or choices. The feasible combinations of complex
  variables have to be stored explicitly in a columnar data format. Depending on the
  feasibility patterns this data format might be further compressed.
  Examples are discrete choice variables that do not have the same options in all
  states, accumulated human capital or wealth if initial levels are zero. In respy,
  complex_variables are called core_dimensions.


Current Implementation
----------------------

- The state space is a dictionary.
- The feasible values of simple variables are stored as a dictionary of grids
- The feasible combinations of all complex state variables are stored as a dict of
  1d arrays (equivalent to a DataFrame). There will be an indexer to map from states
  to positions in the arrays.
- Indices and weights of child states are not pre-computed and thus not part of the
  state space representation. This is done for simplicity and reduction of memory
  requirements.

Open Questions and Limatations
------------------------------

- The choice sets are stored in a very uncompressed way in the state_choice space.
  Might want to do something more similar to respy.
- Filtering is not yet implemented.
- The indexer is not yet implemented.

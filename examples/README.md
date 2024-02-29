# Example model specifications

## Choosing an example

| Example name                        | Description                                       | Runtime       |
| ----------------------------------- | ------------------------------------------------- | ------------- |
| [`long_running`](./long_running.py) | Consumption-savings model with health and leisure | a few minutes |

## Running an example

Say you want to solve the `long_running` example locally. First, clone this repository,
create the `lcm` conda environment, move into the examples folder, and open the
interactive Python shell. In a console, type:

```console
$ git clone https://github.com/OpenSourceEconomics/lcm.git
$ cd lcm
$ conda env create -f environment.yml
$ conda activate lcm
$ cd examples
$ ipython
```

In that shell, run the following code:

```python
from lcm.entry_point import get_lcm_function

from long_running import MODEL_CONFIG, PARAMS


solve_model, _ = get_lcm_function(model=MODEL_CONFIG)
vf_arr = solve_model(PARAMS)
```

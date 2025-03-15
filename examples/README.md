# Example model specifications

## Choosing an example

| Example name                        | Description                                       | Runtime       |
| ----------------------------------- | ------------------------------------------------- | ------------- |
| [`long_running`](./long_running.py) | Consumption-savings model with health and leisure | a few minutes |

## Running an example

Say you want to solve the `long_running` example locally. First, clone this repository,
[install pixi if required](https://pixi.sh/latest/#installation), move into the examples
folder, and open the interactive Python shell. In a console, type:

```console
$ git clone https://github.com/opensourceeconomics/pylcm.git
$ cd lcm/examples
$ pixi run ipython
```

In that shell, run the following code:

```python
from lcm.entry_point import get_lcm_function

from long_running import MODEL_CONFIG, PARAMS


solve_model, _ = get_lcm_function(model=MODEL_CONFIG, targets="solve")
V_arr_list = solve_model(PARAMS)
```

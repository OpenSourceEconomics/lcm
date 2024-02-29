# Example model specifications

## Choosing an example

| Example name   | Description                                       | Runtime       |
| -------------- | ------------------------------------------------- | ------------- |
| `long_running` | Consumption-savings model with health and leisure | a few minutes |

## Running an example

Say you want to solve the [`long_running`](./long_running.py) example locally. First,
clone this repository and move into the example folder. In a console, type:

```console
$ git clone https://github.com/OpenSourceEconomics/lcm.git
$ cd lcm/examples
```

Make sure that you have `lcm` installed in your Python environment. Then, in a Python
shell run

```python
from lcm.entry_point import get_lcm_function

from long_running import MODEL_CONFIG, PARAMS


solve_model, _ = get_lcm_function(model=MODEL_CONFIG)
vf_arr = solve_model(PARAMS)
```

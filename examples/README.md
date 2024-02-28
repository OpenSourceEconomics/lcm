# Example model specifications

## Choosing an example

| Example name   | Description                                        | Runtime       |
| -------------- | -------------------------------------------------- | ------------- |
| `long_running` | Consumptions-savings model with health and leisure | a few minutes |

## Running an example

Say you want to run the [`long_running`](./long_running.py) example locally. In a Python
shell, execute:

```python
from lcm.entry_point import get_lcm_function

from long_running import MODEL_CONFIG, PARAMS


solve_model, _ = get_lcm_function(model=MODEL_CONFIG)
vf_arr = solve_model(PARAMS)
```

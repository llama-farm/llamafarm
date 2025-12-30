# Custom Runtime Demo

This example demonstrates how to use the Custom Runtime engine to run your own Python code as tools within LlamaFarm.

## Contents

- `llamafarm.yaml`: Configuration defining the runtime and pointing to custom code.
- `data_tools.py`: Python module containing functions decorated with `@tool`.
- `sdk.py`: A simple SDK helper (in a real project, this would be imported from `llamafarm`).

## How to Run

1.  **Start LlamaFarm Services**:
    Ensure the LlamaFarm services are running in a separate terminal:
    ```bash
    lf start
    ```

2.  **Run the Demo**:
    Execute the run script which will query the model.
    ```bash
    ./run_demo.sh
    ```

3.  **Expected Output**:
    The model should call `process_data` from `data_tools.py` and report the reversed string and length.

## Tools

### `process_data(data: str)`
Reverses the input string and returns a JSON summary with length.

### `factorial(n: int)`
Calculates factorial recursively. Try asking: "Calculate factorial of 5".

class ModelInitilizationError(Exception):
    """Raised when there is an error in the model initialization."""

    def __init__(self, error_message) -> None:
        """Initialize the exception with a list of errors or a single error message."""
        if isinstance(error_message, list):
            error_message = format_errors(error_message)
        super().__init__(error_message)


class GridInitializationError(Exception):
    """Raised when there is an error in the grid initialization."""

    def __init__(self, error_message) -> None:
        """Initialize the exception with a list of errors or a single error message."""
        if isinstance(error_message, list):
            error_message = format_errors(error_message)
        super().__init__(error_message)


def format_errors(errors: list[str]) -> str:
    """Convert list of error messages into a single string."""
    if len(errors) == 1:
        formatted = errors[0]
    else:
        enumerated = "\n\n".join([f"{i}. {error}" for i, error in enumerate(errors, 1)])
        formatted = f"The following errors occurred:\n\n{enumerated}"
    return formatted

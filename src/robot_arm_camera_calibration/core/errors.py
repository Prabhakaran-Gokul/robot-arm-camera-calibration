class CalibrationError(Exception):
    """Base class for all errors raised by this library."""


class TargetNotDetectedError(CalibrationError):
    """Raised when a sample capture is attempted but the target isn't visible."""


class JogLimitViolationError(CalibrationError):
    """Raised when a requested jog motion would exit the configured workspace bounds."""


class InsufficientSamplesError(CalibrationError):
    """Raised when a solver is run with fewer samples than it requires."""

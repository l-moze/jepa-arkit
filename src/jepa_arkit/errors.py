class ContractError(ValueError):
    """A versioned data or model contract is invalid."""


class TrackViolation(PermissionError):
    """Research-only ancestry was requested from the product track."""


class GateBlocked(RuntimeError):
    """A readiness gate cannot pass with the available evidence."""



class TransientWebhookError(Exception):
    """Raise this when a webhook fails due to temporary infrastructure issues. 
    Signals the SQS consumer to NOT delete the message."""
    pass

class AppError(Exception):
    """Base class for all custom exceptions in this app."""
    pass

class DatabaseOperationError(AppError):
    """Raised when a DB transaction fails (connection, constraint, etc.)"""
    pass

class AuthenticationError(AppError):
    """Raised when auth-related logic fails (invalid token, CSRF, etc.)"""
    pass

class RecordNotFoundError(AppError):
    """Raised when an expected database record is missing."""
    pass
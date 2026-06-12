
class TransientWebhookError(Exception):
    """Raise this when a webhook fails due to temporary infrastructure issues. 
    Signals the SQS consumer to NOT delete the message."""
    pass
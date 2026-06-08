# Helper: parse token expiry from GitHub response
from typing import Optional
from datetime import datetime , timezone
 
def _parse_token_expiry(token_data: dict) -> Optional[datetime]:
    """Parse expires_in (seconds) from GitHub token response."""
    expires_in = token_data.get("expires_in")
    if not expires_in:
        return None
    from datetime import timedelta
    return datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))
 
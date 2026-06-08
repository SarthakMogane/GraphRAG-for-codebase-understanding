import httpx
from authlib.integrations.starlette_client import OAuth
from src.core.config import get_settings 

settings = get_settings()

# ── Authlib Setup ─────────────────────────────────────────────────────────────
oauth = OAuth()

oauth.register(
    name='github',
    client_id=settings.GITHUB_CLIENT_ID,
    client_secret=settings.GITHUB_CLIENT_SECRET,
    code_challenge_method='S256',
    access_token_url='https://github.com/login/oauth/access_token',
    authorize_url='https://github.com/login/oauth/authorize',
    api_base_url='https://api.github.com/',
    client_kwargs={
        'scope': 'user:email',
        'timeout': 10.0,
        'limits': httpx.Limits(max_connections=300, max_keepalive_connections=100)
    }, 
)

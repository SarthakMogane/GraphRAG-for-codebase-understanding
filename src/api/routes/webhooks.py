from fastapi import APIRouter, Request, HTTPException, Header
import logging
from src.services.github import GitHubService
from src.db.mock_db import MOCK_DB

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/webhooks", tags=["Webhooks"])

@router.post("/github")
async def github_webhook(
    request: Request,
    x_hub_signature_256: str = Header(None),
    x_github_event: str = Header(None)
):
    """Listens for GitHub App events (like installs and uninstalls)."""
    
    # 1. SECURITY FIRST: Read the raw bytes of the request body
    payload_bytes = await request.body()
    
    # 2. Cryptographically verify GitHub sent this using your webhook secret
    if not GitHubService.validate_webhook_signature(payload_bytes, x_hub_signature_256):
        logger.error("Webhook signature validation failed!")
        raise HTTPException(status_code=401, detail="Invalid signature")

    # 3. Parse the JSON safely now that we know it's authentic
    payload = await request.json()
    action = payload.get("action")
    
    # 4. Handle "Installation" events
    if x_github_event == "installation":
        installation_id = payload["installation"]["id"]
        
        # Who triggered this? (The GitHub user ID of the person who installed/uninstalled it)
        sender_id = str(payload["sender"]["id"]) 
        
        if action == "created":
            logger.info(f"[WEBHOOK] App installed! ID: {installation_id} by User: {sender_id}")
            # Ensure the user exists in our DB before updating
            if sender_id not in MOCK_DB.get("installations", {}):
                MOCK_DB.setdefault("installations", {})
            
            MOCK_DB["installations"][sender_id] = installation_id
            
        elif action == "deleted":
            logger.info(f"[WEBHOOK] App UNINSTALLED! ID: {installation_id} by User: {sender_id}")
            # Wipe their installation from our database instantly
            if sender_id in MOCK_DB.get("installations", {}):
                del MOCK_DB["installations"][sender_id]

    # Return 200 OK so GitHub knows we received it successfully
    return {"status": "ok"}


# # app/api/routes/webhooks.py

# from fastapi import APIRouter, Request, Header, HTTPException
# import logging

# from src.services.github import GitHubService
# from src.db.mock_db import MOCK_DB

# router = APIRouter(prefix="/webhooks", tags=["Webhooks"])
# logger = logging.getLogger(__name__)

# @router.post("/github")
# async def github_webhook_receiver(
#     request: Request,
#     x_hub_signature_256: str = Header(None),
#     x_github_event: str = Header(None)
# ):
#     """
#     Receives all events from GitHub (Installations, Pushes, PRs).
#     Must verify the cryptographic signature before processing.
#     """
#     # 1. Read the raw bytes (required for signature verification)
#     payload_bytes = await request.body()

#     # 2. Verify Security Signature
#     if not GitHubService.validate_webhook_signature(payload_bytes, x_hub_signature_256):
#         logger.error("Webhook dropped: Invalid cryptographic signature.")
#         raise HTTPException(status_code=401, detail="Invalid signature")

#     # 3. Parse JSON safely now that we know GitHub sent it
#     payload = await request.json()

#     # ------------------------------------------------------------------
#     # EVENT ROUTER
#     # ------------------------------------------------------------------
    
#     if x_github_event == "installation" and payload.get("action") == "created":
#         # A user just installed our App!
#         installation_id = payload["installation"]["id"]
#         github_id = payload["sender"]["id"] # The user who installed it
        
#         # Save to mock database so our background workers can use it
#         MOCK_DB["installations"][github_id] = installation_id
        
#         print(f"[DB MOCK] Linked Installation {installation_id} to User {github_id}")
        
#     elif x_github_event == "push":
#         # Someone pushed code. We would trigger Tree-sitter AST re-indexing here.
#         repo_name = payload["repository"]["full_name"]
#         print(f"[BACKGROUND TASK] Code pushed to {repo_name}. Triggering GraphRAG index.")

#     elif x_github_event == "repository" and payload.get("action") == "deleted":
#         # A user deleted their repo. We should drop it from Neo4j.
#         repo_name = payload["repository"]["full_name"]
#         print(f"[BACKGROUND TASK] Repo {repo_name} deleted. Removing from Graph database.")

#     # GitHub requires a 200 OK response within 10 seconds, or it assumes we crashed.
#     # Always acknowledge receipt quickly.
#     return {"status": "ok", "event": x_github_event}
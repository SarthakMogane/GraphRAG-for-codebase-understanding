from fastapi import APIRouter, Request, HTTPException, Header
import logging
from src.services.github import GitHubService
from src.db.mock_db import MOCK_DB , MockRepository

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

    if x_github_event == "installation_repositories":
        installation_id = str(payload["installation"]["id"])
        action = str(payload.get("action"))

        if action == "added":
            for repo_data in payload.get("repositories_added",[]):
                full_name = repo_data["full_name"]
                github_id = repo_data["id"]

                if full_name not in MOCK_DB["repositories"]:
                    MOCK_DB["repositories"][full_name] = MockRepository(
                        github_id=github_id,
                        full_name=full_name,
                        installation_id=installation_id,
                        status=None,         # Ready for setup!
                        is_stale=False
                    ) 

        elif action == "removed":
            for repo_data in payload.get("repositories_removed", []):
                full_name = repo_data["full_name"]

                # Clean up your database safely since you no longer have access
                if full_name in MOCK_DB["repositories"]:
                    del MOCK_DB["repositories"][full_name]

        elif action  == "rename":
             # 1. Extract the single new full name from the top level
            new_full_name = payload["repository"]["full_name"]
            
            # 2. Extract the old repository name from the "changes" object
            old_name_only = payload["changes"]["repository"]["name"]["from"]
            owner_login = payload["repository"]["owner"]["login"]
            old_full_name = f"{owner_login}/{old_name_only}"

            # 3. Swap the key instantly inside your MOCK_DB without any loops
            mock_repo_tables = MOCK_DB["repositories"]
            if old_full_name in mock_repo_tables:
                repo_obj = mock_repo_tables[old_full_name]
                
                # Update the internal attribute
                repo_obj.full_name = new_full_name 
                
                # Assign to new dictionary key and delete the old key
                mock_repo_tables[new_full_name] = repo_obj 
                del mock_repo_tables[old_full_name]

    # Return 200 OK so GitHub knows we received it successfully
    return {"status": "ok"}


import aws_cdk as cdk
from constructs import Construct
from graphrag_infra.constructs.queues import QueuesConstruct
from graphrag_infra.constructs.storage import StorageConstruct
from graphrag_infra.constructs.security import SecurityConstruct

class GraphRagInfrastructureStack(cdk.Stack):

    def __init__(self, scope: Construct, construct_id: str, env_modifier: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # 1. Provision Messaging Layer (SQS)
        # messaging = QueuesConstruct(self, "MessagingLayer", env_modifier=env_modifier)

        # 2. Provision Storage & Artifact Layer (S3 & ECR)
        storage = StorageConstruct(self, "StorageLayer", env_modifier=env_modifier)

        # 3. Provision Security & Permission Layer (IAM)
        # Pass references of the resources down to wire up permissions
        security = SecurityConstruct(
            self, "SecurityLayer",
            env_modifier=env_modifier,
            storage_bucket=storage.storage_bucket,
            base_ecr_repo=storage.base_ecr_repo
        )

        # Output properties to screen upon deployment completion
        cdk.CfnOutput(self, "BucketName", value=storage.storage_bucket.bucket_name)
        cdk.CfnOutput(self, "EcrRepoUri", value=storage.base_ecr_repo.repository_uri)
        # cdk.CfnOutput(self, "IngestionQueueUrl", value=messaging.ingestion_queue.queue_url)

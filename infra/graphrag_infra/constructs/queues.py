import aws_cdk as cdk
from aws_cdk import aws_sqs as sqs
from constructs import Construct

class QueuesConstruct(Construct):

    def __init__(self, scope: Construct, id: str, env_modifier: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        self.webhook_queue = sqs.Queue(
            self,
            "WebhookQueue",
            queue_name=f"graphrag-webhook-queue-{env_modifier}",
            visibility_timeout=cdk.Duration.seconds(300),
        )

        self.ingestion_queue = sqs.Queue(
            self,
            "IngestionQueue",
            queue_name=f"graphrag-ingestion-queue-{env_modifier}",
            visibility_timeout=cdk.Duration.seconds(900),
        )

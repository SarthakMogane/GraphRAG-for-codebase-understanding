import aws_cdk as cdk
from aws_cdk import aws_ecr as ecr, aws_iam as iam, aws_s3 as s3, aws_sqs as sqs
from constructs import Construct

class SecurityConstruct(Construct):

    def __init__(
        self, 
        scope: Construct, 
        id: str, 
        env_modifier: str,
        storage_bucket: s3.IBucket,
        base_ecr_repo: ecr.IRepository,
        webhook_queue: sqs.IQueue,
        ingestion_queue: sqs.IQueue,
        **kwargs
    ) -> None:
        super().__init__(scope, id, **kwargs)

        # Corrected Trust Policy Principal (Changed from broken :// string)
        self.build_role = iam.Role(
            self,
            "MicroVMBuildRole",
            role_name=f"MicroVMBuildRole-{env_modifier}",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"), 
        )

        # Wire up Least Privilege Actions
        base_ecr_repo.grant_pull(self.build_role)
        storage_bucket.grant_read_write(self.build_role)
        webhook_queue.grant_consume_messages(self.build_role)
        ingestion_queue.grant_consume_messages(self.build_role)

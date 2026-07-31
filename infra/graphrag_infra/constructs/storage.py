import aws_cdk as cdk
from aws_cdk import aws_ecr as ecr, aws_s3 as s3
from constructs import Construct

class StorageConstruct(Construct):

    def __init__(self, scope: Construct, id: str, env_modifier: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        stack = cdk.Stack.of(self)

        self.storage_bucket = s3.Bucket(
            self,
            "StorageBucket",
            bucket_name=f"graphrag-deployments-{stack.account}-{stack.region}",
            removal_policy=cdk.RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            enforce_ssl=True,
        )

        self.base_ecr_repo = ecr.Repository(
            self,
            "BaseEcrRepo",
            repository_name=f"graphrag-sandbox-base-{env_modifier}", # Added modifier to avoid name clashes
            removal_policy=cdk.RemovalPolicy.DESTROY,
            image_scan_on_push=True,
        )

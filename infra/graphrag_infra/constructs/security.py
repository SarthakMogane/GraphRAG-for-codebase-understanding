import aws_cdk as cdk
from aws_cdk import aws_ecr as ecr, aws_iam as iam, aws_s3 as s3
from constructs import Construct

class SecurityConstruct(Construct):

    def __init__(
        self, 
        scope: Construct, 
        id: str, 
        env_modifier: str,
        storage_bucket: s3.IBucket,
        base_ecr_repo: ecr.IRepository,
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
        

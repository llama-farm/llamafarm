from fastapi import APIRouter, Query
from services.health_service import health_summary


router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def get_health(cli_client: bool = Query(False, description="Set to true for CLI client requests")):
    return health_summary(cli_client=cli_client)


@router.get("/liveness")
def get_liveness():
    return {"status": "alive"}

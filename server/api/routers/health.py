from fastapi import APIRouter
from services.health_service import health_summary


router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def get_health():
    return health_summary()


@router.get("/liveness")
def get_liveness():
    return {"status": "alive"}

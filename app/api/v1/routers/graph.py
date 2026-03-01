from fastapi import APIRouter

router = APIRouter(prefix="/graph", tags=["Graph"])


@router.post("/start")
def start_graph():
    pass

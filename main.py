from fastapi import FastAPI

try:
    from server_conversation import app as conversation_app
    from server_subtitle import app as subtitle_app
    from sync_api import router as sync_router
except ModuleNotFoundError:
    from face_server.server_conversation import app as conversation_app
    from face_server.server_subtitle import app as subtitle_app
    from face_server.sync_api import router as sync_router

app = FastAPI(title="face_server")


@app.get("/")
async def root():
    return {
        "status": "ok",
        "routes": {
            "conversation_ws": "/conversation/ws",
            "subtitle_ws": "/subtitle/ws",
            "sync_health": "/sync/health",
            "sync_sessions": "/sync/sessions",
        },
    }


app.include_router(sync_router, prefix="/sync")
app.mount("/conversation", conversation_app)
app.mount("/subtitle", subtitle_app)

from fastapi import FastAPI

try:
    from server import app as conversation_app
    from server_subtitle import app as subtitle_app
except ModuleNotFoundError:
    from face_server.server import app as conversation_app
    from face_server.server_subtitle import app as subtitle_app

app = FastAPI(title="face_server")


@app.get("/")
async def root():
    return {
        "status": "ok",
        "routes": {
            "conversation_ws": "/conversation/ws",
            "subtitle_ws": "/subtitle/ws",
        },
    }


app.mount("/conversation", conversation_app)
app.mount("/subtitle", subtitle_app)

def main() -> None:
    import uvicorn
    uvicorn.run("elder_guard.web.app:app", host="127.0.0.1", port=8000)
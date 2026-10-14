from fastapi import FastAPI

app = FastAPI()

from router.application import application

app.include_router(application)
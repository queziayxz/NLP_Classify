from fastapi import APIRouter, Request, HTTPException
from fastapi.templating import Jinja2Templates
from pathlib import Path
from pydantic import BaseModel, Field
from app.model import Model
from app.mutation_phrase import MutationPhrase

application = APIRouter(prefix="/app", tags=["application"])

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=BASE_DIR/"templates")

model = Model()
mutation = MutationPhrase()

class InputModel(BaseModel):
    texto: str = Field(min_length=20, max_length=60_000)

@application.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html"
    )

@application.get("/model")
async def page_model(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="model.html"
    )

@application.get("/mutation")
async def page_mutation(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="mutation.html"
    )

@application.post("/classify")
async def classify(payload: InputModel):
    result = model.predict(text=payload.texto);
    return result

@application.post("/synonyms")
async def synonyms(payload: InputModel):
    try:
        result = mutation.swap_synonyms(phrase=payload.texto)
    except Exception as error:
        raise HTTPException(status_code=402, detail=str(error))
    return {"result":result}

@application.post("/parafrase")
async def parafrase(payload: InputModel):
    result = mutation.parafrasear(phrase=payload.texto)
    return {"result":result}

@application.post("/combinacao")
async def combinacao(payload: InputModel):
    try:
        result = mutation.mutation(phrase=payload.texto)
    except Exception as error:
        raise HTTPException(status_code=402, detail=str(error))
    return {"result":result}
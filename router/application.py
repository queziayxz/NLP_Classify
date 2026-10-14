from fastapi import APIRouter, Request
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
    # if hasattr(result, "tolist"):
    #     result = result.tolist()
    return result

@application.post("/synonyms")
async def synonyms(payload: InputModel):
    result = mutation.swap_synonyms(phrase=payload.texto)
    print(result)
    return {"result":result}
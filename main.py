# main.py
from fastapi import FastAPI, HTTPException, UploadFile, File   # HTTPException = proper error codes; UploadFile/File = image uploads
from pydantic import BaseModel                                 # for validating request bodies

from app.bigram_model import BigramModel       # bigram logic
from app.embeddings import EmbeddingModel      # spaCy embedding logic
from app.classifier import predict             # CNN image classification

app = FastAPI()

# Bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]
bigram_model = BigramModel(corpus)

# Embedding model
embedding_model = EmbeddingModel("en_core_web_lg")


# Request schemas
class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    word: str                                  # the query word to embed


class SimilarityRequest(BaseModel):
    word1: str
    word2: str


# Endpoints 
@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    # reject empty input 
    if not request.word.strip():
        raise HTTPException(status_code=422, detail="word must not be empty")
    # reject words spaCy has no vector for 
    if not embedding_model.has_vector(request.word):
        raise HTTPException(status_code=404, detail="No embedding found for this word")

    vector = embedding_model.get_embedding(request.word)
    return {
        "word": request.word,
        "dimension": len(vector),
        "embedding": vector,
    }


@app.post("/similarity")                       # bonus endpoint
def get_similarity(request: SimilarityRequest):
    score = embedding_model.get_similarity(request.word1, request.word2)
    return {"word1": request.word1, "word2": request.word2, "similarity": score}


# Image classifier 
# POST /classify: upload an image file, get back the predicted CIFAR10 class
@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    # Reject anything that isn't an image
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    # Read the uploaded bytes and hand them to the model
    image_bytes = await file.read()
    try:
        result = predict(image_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")

    return {"filename": file.filename, **result}
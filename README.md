# sps_genai

FastAPI app with a bigram text generator, spaCy word embeddings, and a CNN image classifier.

## Run with Docker
```bash
docker build -t asm2-genai .
docker run --rm -p 8000:80 asm2-genai
```
Docs: http://127.0.0.1:8000/docs

## Endpoints
- `POST /generate`: `{"start_word": "bigram", "length": 5}`
- `POST /embedding`: `{"word": "apple"}` returns a 300-dimension vector
- `POST /similarity`: `{"word1": "apple", "word2": "orange"}`
- `POST /classify`: upload an image file, returns the predicted CIFAR10 class

## Examples
```bash
curl -X POST http://127.0.0.1:8000/embedding \
  -H "Content-Type: application/json" \
  -d '{"word": "apple"}'

curl -X POST http://127.0.0.1:8000/classify -F "file=@your_image.png"
```

## Assignment 2: CNN image classifier

`POST /classify` accepts an image upload and returns the predicted CIFAR10 class
(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) with probabilities.

Model: `app/cnn_model.py` (Conv16 -> Pool -> Conv32 -> Pool -> FC100 -> FC10, 64x64x3 input).
Trained with `train_cnn.py` on CIFAR10 (resized to 64x64); weights saved in `app/cnn_cifar10.pth`.
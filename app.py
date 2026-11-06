import io
import os

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps

TAMANO = 224
BASE = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()

_interprete = None
_etiquetas = []


def cargar_modelo():
    """Carga el modelo la primera vez que se predice (Render arranca rápido así)."""
    global _interprete, _etiquetas
    if _interprete is not None:
        return

    # ai-edge-litert es el sucesor de tflite-runtime: soporta modelos exportados
    # por versiones nuevas de TensorFlow (tflite-runtime 2.14 falla con FULLY_CONNECTED v12).
    from ai_edge_litert.interpreter import Interpreter

    with open(os.path.join(BASE, "modelo.tflite"), "rb") as archivo:
        _interprete = Interpreter(model_content=archivo.read())
    _interprete.allocate_tensors()

    with open(os.path.join(BASE, "etiquetas.txt"), encoding="utf-8") as archivo:
        _etiquetas = [linea.strip() for linea in archivo if linea.strip()]


def preprocesar(imagen: Image.Image) -> np.ndarray:
    """Recorte central cuadrado + resize, con píxeles 0..255 como espera el modelo."""
    # Las fotos de celular suelen venir rotadas por EXIF; sin esto llegan acostadas.
    imagen = ImageOps.exif_transpose(imagen).convert("RGB")
    lado = min(imagen.size)
    izquierda = (imagen.width - lado) // 2
    arriba = (imagen.height - lado) // 2
    recorte = imagen.crop((izquierda, arriba, izquierda + lado, arriba + lado))
    recorte = recorte.resize((TAMANO, TAMANO))
    return np.asarray(recorte, dtype=np.float32)[None, ...]


@app.get("/")
async def raiz():
    return {"estado": "ok", "clases": _etiquetas}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        cargar_modelo()

        imagen = Image.open(io.BytesIO(await file.read()))
        entrada = preprocesar(imagen)

        _interprete.set_tensor(_interprete.get_input_details()[0]["index"], entrada)
        _interprete.invoke()
        salida = _interprete.get_tensor(_interprete.get_output_details()[0]["index"])[0]

        indice = int(np.argmax(salida))
        return {
            "prediction": _etiquetas[indice],
            "confianza": float(salida[indice]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

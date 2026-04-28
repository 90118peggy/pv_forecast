import os
import sys

from fastapi import FastAPI

if __package__ is None or __package__ == '':
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.routes.predictions import router as prediction_router

app = FastAPI(title='PV Forecast API')

app.include_router(prediction_router)

@app.get('/health')
def health():
    return {'status': 'ok'}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run('app.main:app', host='127.0.0.1', port=8000, reload=True)

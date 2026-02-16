from fastapi import FastAPI
import uvicorn 

from api import all_routers


app = FastAPI()

for router in all_routers:
    app.include_router(router)


if __name__ == "__main__":
    uvicorn.run('main:app', reload=True)
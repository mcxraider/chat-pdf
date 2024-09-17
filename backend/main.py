from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for all origins (you can restrict this to specific domains)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


@app.get("/data/")
async def get_data():
    # Example data to return as JSON
    data = {
        "message": "Hello, World!",
        "status": "success",
        "data": {
            "item_1": "Value 1",
            "item_2": "Value 2",
        },
    }
    return JSONResponse(content=data)


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()  # Read the file content (if needed)
    # Here, you could save the file or process it as needed
    return JSONResponse(
        content={"filename": file.filename, "message": "File uploaded successfully!"}
    )

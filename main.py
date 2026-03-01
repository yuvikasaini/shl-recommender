from fastapi import FastAPI,Query
from pydantic import BaseModel
from vector_store import SHLVectorStore
from recommender import SHLRecommender
import uvicorn

app = FastAPI(title="SHL Smart Recommender API")

# Setup on start
store = SHLVectorStore()
store.build_index()
recommender = SHLRecommender(store)

class JDRequest(BaseModel):
    text: str

@app.post("/recommend")
async def recommend(request: JDRequest):
    # The 3-Stage Advanced Pipeline: 
    # 1. Hybrid Search -> 2. Balanced MMR -> 3. LLM Selection
    result = recommender.get_final_recommendations(request.text)
    return result

@app.get("/search")
async def search_endpoint(query: str = Query(..., description="The job search text")):
    """
    GET endpoint that takes a search string and returns JSON results.
    Example: /search?query=data scientist
    """
    # This calls your existing recommendation function
    results = recommender.get_final_recommendations(query)
    return {
        "status": "success",
        "query_received": query,
        "results": results
    }



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
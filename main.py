import asyncio
import os
import httpx
from fastapi import FastAPI, HTTPException, Body
from typing import Optional, List
from pydantic import BaseModel
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()

SERVICE_TITLE = os.getenv("SERVICE_TITLE", "Seedream 5 Lite Service")
app = FastAPI(title=SERVICE_TITLE)

RH_API_KEY = os.getenv("RUNNINGHUB_API_KEY")

async def get_image_dimensions(image_url: str):
    """Download image header and determine width and height."""
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            headers = {"Range": "bytes=0-10240"}
            resp = await client.get(image_url, headers=headers)
            if resp.status_code not in [200, 206]:
                resp = await client.get(image_url) 
            
            img = Image.open(io.BytesIO(resp.content))
            return img.size
    except Exception as e:
        print(f"ERROR - Dimension detection failed: {e}")
        return (1024, 1024)

def calculate_max_dimensions_for_aspect_ratio(width: int, height: int):
    """
    Calculate the MAXIMUM dimensions for the detected aspect ratio,
    within RunningHub constraints (1600-4704 width, 1344-4096 height).
    """
    MIN_W, MAX_W = 1600, 4704
    MIN_H, MAX_H = 1344, 4096
    
    aspect_ratio = width / height
    
    valid_ratios = [
        (1.0, "1:1"),
        (16/9, "16:9"),
        (9/16, "9:16"),
        (4/3, "4:3"),
        (3/4, "3:4"),
        (3/2, "3:2"),
        (2/3, "2:3"),
        (5/4, "5:4"),
        (4/5, "4:5"),
    ]
    
    closest_ratio, closest_name = min(valid_ratios, key=lambda x: abs(x[0] - aspect_ratio))
    
    print(f"Detected aspect ratio: {aspect_ratio:.3f} -> Using {closest_name}")
    
    if closest_name == "1:1":
        new_width = min(MAX_W, 4096)
        new_height = new_width
    elif closest_name == "16:9":
        new_width = MAX_W
        new_height = int(MAX_W / 16 * 9)
    elif closest_name == "9:16":
        new_height = MAX_H
        new_width = int(MAX_H / 16 * 9)
    elif closest_name == "4:3":
        new_width = MAX_W
        new_height = int(MAX_W / 4 * 3)
    elif closest_name == "3:4":
        new_height = MAX_H
        new_width = int(MAX_H / 4 * 3)
    elif closest_name == "3:2":
        new_width = MAX_W
        new_height = int(MAX_W / 3 * 2)
    elif closest_name == "2:3":
        new_height = MAX_H
        new_width = int(MAX_H / 3 * 2)
    elif closest_name == "5:4":
        new_width = MAX_W
        new_height = int(MAX_W / 5 * 4)
    elif closest_name == "4:5":
        new_height = MAX_H
        new_width = int(MAX_H / 5 * 4)
    else:
        new_width = MAX_W
        new_height = MAX_H
    
    new_width = (new_width // 16) * 16
    new_height = (new_height // 16) * 16
    
    new_width = max(MIN_W, min(MAX_W, new_width))
    new_height = max(MIN_H, min(MAX_H, new_height))
    
    return new_width, new_height, closest_name

class SeedreamRequest(BaseModel):
    image_url: str
    reference_image_urls: Optional[List[str]] = []
    prompt: str
    max_images: Optional[int] = 1
    apiKey: Optional[str] = None

@app.get("/v1/health")
async def health():
    return {"status": "ok", "service": "seedream5-lite"}

@app.post("/v1/execute-seedream")
async def execute_seedream(req: SeedreamRequest):
    api_key = req.apiKey or RH_API_KEY
    if not api_key:
        raise HTTPException(status_code=500, detail="RunningHub API Key not configured")

    # 1. Prepare image URLs array (Main + References)
    image_urls = [req.image_url]
    if req.reference_image_urls:
        image_urls.extend(req.reference_image_urls)
    
    # Max 10 images supported
    if len(image_urls) > 10:
        image_urls = image_urls[:10]

    # 2. Detect dimensions and calculate MAX size for that aspect ratio
    width, height = await get_image_dimensions(req.image_url)
    
    output_width, output_height, aspect_name = calculate_max_dimensions_for_aspect_ratio(width, height)
    
    print(f"Input: {width}x{height} ({width/height:.3f})")
    print(f"Output: MAX {output_width}x{output_height} for aspect ratio {aspect_name}")

    payload = {
        "imageUrls": image_urls,
        "prompt": req.prompt,
        "width": output_width,
        "height": output_height,
        "maxImages": req.max_images,
        "sequentialImageGeneration": "auto"
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    print(f"Starting Seedream 5 Lite task for {req.image_url}...")
    
    # 3. Start Task
    async with httpx.AsyncClient(timeout=60.0) as client:
        url = "https://www.runninghub.ai/openapi/v2/seedream-v5-lite/image-to-image"
        try:
            resp = await client.post(url, headers=headers, json=payload)
            res_data = resp.json()
        except httpx.RequestError as exc:
            raise HTTPException(status_code=504, detail=f"Connection Error to RunningHub: {str(exc)}")

        if resp.status_code != 200 or not res_data.get("taskId"):
            raise HTTPException(status_code=502, detail=f"Failed to start Seedream task: {res_data}")

        task_id = res_data["taskId"]
        
        # 4. Polling for Success
        query_url = "https://www.runninghub.ai/openapi/v2/query"
        max_retries = 60 # 5 minutes
        
        for _ in range(max_retries):
            await asyncio.sleep(5)
            try:
                poll_resp = await client.post(query_url, headers=headers, json={"taskId": task_id})
                poll_data = poll_resp.json()
            except httpx.RequestError:
                continue 
                
            status = poll_data.get("status")
            if status == "SUCCESS":
                results = poll_data.get("results", [])
                if results:
                    return {
                        "status": "success",
                        "taskId": task_id,
                        "output_url": results[0].get("url"),
                        "all_results": [r.get("url") for r in results]
                    }
                else:
                    raise HTTPException(status_code=500, detail="Task succeeded but no results")
            
            elif status in ["FAILED", "EXCEPTION", "CANCELLED"]:
                error_msg = poll_data.get("errorMessage", "Unknown error")
                raise HTTPException(status_code=502, detail=f"Generation failed: {error_msg}")
        
        raise HTTPException(status_code=504, detail="Timeout waiting for Seedream task")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

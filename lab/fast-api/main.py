from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from models import ItemPayload

app = FastAPI()
grocery_list: dict[int, ItemPayload] = {}
favicon_path: str = 'favicon.ico'

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)

@app.get("/")
def root() -> dict[str, str]:
    return { "message": "Hello World" }

@app.post("/items/{item_name}/{quantity}")
def add_item(item_name: str, quantity: int) -> dict[str, ItemPayload]:
    if quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be greater than 0.")
    items_ids: dict[str, int] = {
        item.item_name: item.item_id if item.item_id is not None else 0
        for item in grocery_list.values()
    }
    if item_name in items_ids.keys():
        item_id: int = items_ids[item_name]
        grocery_list[item_id].quantity += quantity
    else:
        item_id: int = max(grocery_list.keys()) + 1 if grocery_list else 0
        grocery_list[item_id] = ItemPayload(
            item_id=item_id, item_name=item_name, quantity=quantity
        )
    
    return { "item": grocery_list[item_id] }

@app.get("/items/{item_id}")
def list_item(item_id: int) -> dict[str, ItemPayload]:
    if item_id not in grocery_list:
        raise HTTPException(status_code=404, detail="Item not found.")
    return { "item": grocery_list[item_id] }

@app.get("/items")
def list_items() -> dict[str, dict[int, ItemPayload]]:
    return { "items": grocery_list }

@app.delete("/items/{item_id}")
def delete_item(item_id: int) -> dict[str, str]:
    if item_id not in grocery_list:
        raise HTTPException(status_code=404, detail="Item not found.")
    del grocery_list[item_id]
    return { "result": "Item deleted." }

@app.delete("/items/{item_id}/{quantity}")
def remove_quantity(item_id: int, quantity: int) -> dict[str, str]:
    if item_id not in grocery_list:
        raise HTTPException(status_code=404, detail="Item not found.")
    if grocery_list[item_id].quantity <= quantity:
        del grocery_list[item_id]
        return { "result": "Item deleted." }
    else:
        grocery_list[item_id].quantity -= quantity
    return { "result": f"{quantity} items removed." }
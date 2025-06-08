from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from functioncaller.database import engine, get_db
from functioncaller import model
import uvicorn

# Create database tables
model.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Immune File(Copy Hai JI💀💀)."}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"} # Returns the health status of the API

# Example GET endpoint
@app.get("/api/items")
async def get_items(db: Session = Depends(get_db)):
    try:
        items = db.query(model.Item).all() # Query all items from database
        return items
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) # Handle any database errors

# Example POST endpoint
@app.post("/api/items")
async def create_item(item: model.ItemCreate, db: Session = Depends(get_db)):
    try:
        db_item = model.Item(**item.dict()) # Create new item from request data
        db.add(db_item) # Add item to database
        db.commit() # Commit the transaction
        db.refresh(db_item) # Refresh the item to get generated fields
        return db_item
    except Exception as e:
        db.rollback() # Rollback on error
        raise HTTPException(status_code=500, detail=str(e))

# Example GET by ID endpoint
@app.get("/api/items/{item_id}")
async def get_item(item_id: int, db: Session = Depends(get_db)):
    try:
        item = db.query(model.Item).filter(model.Item.id == item_id).first() # Query item by ID
        if item is None:
            raise HTTPException(status_code=404, detail="Item not found") # Handle not found case
        return item
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Example PUT endpoint
@app.put("/api/items/{item_id}")
async def update_item(item_id: int, item: model.ItemUpdate, db: Session = Depends(get_db)):
    try:
        db_item = db.query(model.Item).filter(model.Item.id == item_id).first() # Get existing item
        if db_item is None:
            raise HTTPException(status_code=404, detail="Item not found")
        for key, value in item.dict(exclude_unset=True).items():
            setattr(db_item, key, value) # Update item attributes
        db.commit() # Commit changes
        db.refresh(db_item) # Refresh the item
        return db_item
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# Example DELETE endpoint
@app.delete("/api/items/{item_id}")
async def delete_item(item_id: int, db: Session = Depends(get_db)):
    try:
        db_item = db.query(model.Item).filter(model.Item.id == item_id).first() # Get item to delete
        if db_item is None:
            raise HTTPException(status_code=404, detail="Item not found")
        db.delete(db_item) # Delete the item
        db.commit() # Commit the deletion
        return {"message": "Item deleted successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)






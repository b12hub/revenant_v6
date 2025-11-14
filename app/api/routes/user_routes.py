from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from db.database import get_db
from app import crud, schemas

router = APIRouter(prefix="/users", tags=["Users"])

@router.post("/", response_model=schemas.user.UserOut)
def create_user(user: schemas.user.UserCreate, db: Session = Depends(get_db)):
    return crud.user_crud.create_user(db, user)

@router.get("/{user_id}", response_model=schemas.user.UserOut)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = crud.user_crud.get_user(db, user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@router.get("/", response_model=list[schemas.user.UserOut])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return crud.user_crud.get_users(db, skip, limit)

@router.put("/{user_id}", response_model=schemas.user.UserOut)
def update_user(user_id: int, user: schemas.user.UserUpdate, db: Session = Depends(get_db)):
    return crud.user_crud.update_user(db, user_id, user)

@router.delete("/{user_id}", response_model=schemas.user.UserOut)
def delete_user(user_id: int, db: Session = Depends(get_db)):
    return crud.user_crud.delete_user(db, user_id)

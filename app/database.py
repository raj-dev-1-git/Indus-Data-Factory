from pathlib import Path

from sqlalchemy import create_engine

from sqlalchemy.orm import sessionmaker, declarative_base

BASE = Path(__file__).resolve().parent.parent


DB_DIR = BASE / "database"

DB_DIR.mkdir(exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_DIR / 'app.db'}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

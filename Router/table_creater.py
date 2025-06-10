from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv



#have to create a .env file with all these credentials
    # DB_USER=
    # DB_PASSWORD=
    # DB_HOST=
    # DB_NAME=

#example:
# DB_USER=postgres
# DB_PASSWORD=12345
# DB_HOST=localhost
# DB_NAME=pattern_detection

# Load environment variables from .env file
load_dotenv()

# Use SQLite for development/testing
# For production, you can switch back to PostgreSQL by setting the environment variables
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")

# If PostgreSQL credentials are provided, use PostgreSQL, otherwise use SQLite
if all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
    SQLALCHEMY_DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
else:
    # Use SQLite for development
    SQLALCHEMY_DATABASE_URL = "sqlite:///./copy_hai_ji.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in SQLALCHEMY_DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

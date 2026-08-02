"""Run once to create all tables: python scripts/init_db.py"""
from cascade.storage.db import init_db

if __name__ == "__main__":
    engine = init_db()
    print(f"Database initialized at: {engine.url}")

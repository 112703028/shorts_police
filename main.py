import uvicorn
from line_bot import app
from database import init_db

if __name__ == "__main__":
    init_db()  # 確保 SQLite 的三張表存在
    uvicorn.run(app, host="0.0.0.0", port=8000)

from flask import Flask

import threading
import time

from routes import register_routes

from camera import (
    camera_thread,
    detect_thread
)

app = Flask(__name__)

register_routes(app)

if __name__ == "__main__":

    print("Starting threads...")

    threading.Thread(
        target=camera_thread,
        daemon=True
    ).start()

    time.sleep(0.3)

    threading.Thread(
        target=detect_thread,
        daemon=True
    ).start()

    print("Server: http://localhost:5000")

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True
    )
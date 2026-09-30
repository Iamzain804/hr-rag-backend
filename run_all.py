import os
import signal
import subprocess
import sys
import time

# Ensure clean UTF-8 on Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SERVICES = [
    {
        "name": "Identity & Org Service",
        "dir": os.path.join(BASE_DIR, "identity-org-service"),
        "port": 8001,
        "cmd": [sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8001", "--reload"],
    },
    {
        "name": "Notification Service",
        "dir": os.path.join(BASE_DIR, "notification-service"),
        "port": 8002,
        "cmd": [sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8002", "--reload"],
    },
    {
        "name": "Ingestion & Vector Service",
        "dir": os.path.join(BASE_DIR, "ingestion-service"),
        "port": 8003,
        "cmd": [sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8003", "--reload"],
    },
    {
        "name": "RAG Chat Service",
        "dir": os.path.join(BASE_DIR, "rag-chat-service"),
        "port": 8004,
        "cmd": [sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8004", "--reload"],
    },
]

def main():
    print("================================================================================")
    print("STARTING ALL BACKEND MICROSERVICES (PORTS: 8001, 8002, 8003, 8004)")
    print("================================================================================")

    processes = []

    try:
        for s in SERVICES:
            print(f" -> Starting [{s['name']}] on http://127.0.0.1:{s['port']}...")
            p = subprocess.Popen(
                s["cmd"],
                cwd=s["dir"],
                shell=False,
            )
            processes.append((s["name"], p))
            time.sleep(1.0)

        print("\n================================================================================")
        print("ALL 4 BACKEND SERVICES ARE NOW RUNNING!")
        print(" - Identity & Org: http://127.0.0.1:8001/docs")
        print(" - Notifications:  http://127.0.0.1:8002/docs")
        print(" - Ingestion:      http://127.0.0.1:8003/docs")
        print(" - RAG Chat:       http://127.0.0.1:8004/docs")
        print("Press Ctrl+C anytime to stop all services simultaneously.")
        print("================================================================================\n")

        # Keep parent process alive to monitor children
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Stopping all backend services gracefully...")
        for name, p in processes:
            print(f" - Terminating {name}...")
            p.terminate()
            try:
                p.wait(timeout=3)
            except Exception:
                p.kill()
        print("[SHUTDOWN] All backend services stopped.")

if __name__ == "__main__":
    main()

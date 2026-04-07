"""Eagle Vision — Single-Entry Production System Launcher

Runs the entire system from one command:
    python main.py
    python main.py --video input/sample5.mp4
    python main.py --no-kafka

Starts: Docker → FastAPI → CV Pipeline → Streamlit UI → Browser
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
DOCKER_COMPOSE = os.path.join(ROOT, "docker-compose.yml")

processes: list[subprocess.Popen] = []
pipeline_proc: subprocess.Popen | None = None
shutting_down = False


def kill_proc_tree(proc: subprocess.Popen):
    """Kill a process and all its children (Windows-safe)."""
    try:
        import psutil
        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
        for child in children:
            child.terminate()
        parent.terminate()
        _, alive = psutil.wait_procs(children + [parent], timeout=5)
        for p in alive:
            p.kill()
    except (ImportError, psutil.NoSuchProcess):
        # psutil not available — fallback to taskkill on Windows
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            try:
                proc.kill()
            except OSError:
                pass


def shutdown(signum=None, frame=None):
    global shutting_down
    if shutting_down:
        return
    shutting_down = True

    print("\n🛑 Shutdown initiated...")

    # Step 1: WAIT for pipeline to finish (writes analytics.json in finally block)
    if pipeline_proc is not None and pipeline_proc.poll() is None:
        print("   ⏳ Waiting for pipeline to finish (max 30s)...")
        try:
            pipeline_proc.wait(timeout=30)
            print("   ✅ Pipeline finished gracefully.")
        except subprocess.TimeoutExpired:
            print("   ⚠️ Pipeline did not exit in time. Force killing...")
            kill_proc_tree(pipeline_proc)

    # Step 3: NOW shutdown everything else (API, DB writer, Streamlit)
    print("   🔻 Shutting down API / DB writer / UI...")
    for proc in reversed(processes):
        if proc is not pipeline_proc and proc.poll() is None:
            kill_proc_tree(proc)

    # Step 4: Final check — force-kill anything still alive
    for proc in processes:
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass


    print("✅ All services stopped.")


def spawn(cmd: list[str], **kwargs) -> subprocess.Popen:
    proc = subprocess.Popen(cmd, cwd=ROOT, **kwargs)
    processes.append(proc)
    return proc


def wait_for_port(port: int, timeout: int = 30):
    import socket
    start = time.time()
    while time.time() - start < timeout:
        try:
            s = socket.create_connection(("localhost", port), timeout=2)
            s.close()
            return True
        except OSError:
            time.sleep(1)
    return False


def main():
    parser = argparse.ArgumentParser(description="Eagle Vision — System Launcher")
    parser.add_argument("--video", default="input/sample.mp4", help="Path to input video")
    parser.add_argument("--model", default="models/best.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.6, help="Detection confidence")
    parser.add_argument("--no-kafka", action="store_true", help="Skip Kafka/Docker")
    parser.add_argument("--no-save", action="store_true", help="Skip saving output video")
    parser.add_argument("--no-docker", action="store_true", help="Skip docker-compose (assume running)")
    args = parser.parse_args()

    video_path = args.video
    if not os.path.isabs(video_path):
        video_path = os.path.join(ROOT, video_path)
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)


    py = sys.executable
    use_kafka = not args.no_kafka

    print("=" * 55)
    print("  🚜 Eagle Vision — Production System Launcher")
    print("=" * 55)
    print(f"  Video:  {args.video}")
    print(f"  Kafka:  {'ON' if use_kafka else 'OFF'}")
    print("=" * 55)

    # ── Step 1: Docker infrastructure ───────────────────────
    if use_kafka and not args.no_docker:
        print("\n🐳 Starting Docker services (Kafka + TimescaleDB)...")
        result = subprocess.run(
            ["docker", "compose", "-f", DOCKER_COMPOSE, "up", "-d"],
            capture_output=True, text=True, cwd=ROOT,
        )
        if result.returncode != 0:
            print(f"   ⚠️  Docker error: {result.stderr.strip()}")
            print("   Is Docker Desktop running?")
            sys.exit(1)
        print("   Waiting for Kafka...", end=" ", flush=True)
        if wait_for_port(9092):
            print("✅")
        else:
            print("⚠️ timeout (continuing)")
        print("   Waiting for TimescaleDB...", end=" ", flush=True)
        if wait_for_port(5432):
            print("✅")
        else:
            print("⚠️ timeout (continuing)")

    # ── Step 2: FastAPI server ──────────────────────────────
    print("\n🚀 Starting API server (port 8000)...")
    spawn([py, "-m", "uvicorn", "kafka_consumers.api_server:app",
           "--host", "0.0.0.0", "--port", "8000", "--log-level", "warning"])
    print("   Waiting for API...", end=" ", flush=True)
    if wait_for_port(8000, timeout=10):
        print("✅")
    else:
        print("⚠️ timeout (continuing)")

    # ── Step 3: DB writer (if Kafka) ────────────────────────
    if use_kafka:
        print("💾 Starting DB writer...")
        spawn([py, "-m", "kafka_consumers.db_writer"],
              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ── Step 4: CV Pipeline (OpenCV window) ───────────────────
    print(f"\n🎥 Starting Pipeline: {args.video}")
    pipeline_cmd = [
        py, "run_pipeline.py",
        "--video", video_path,
        "--model", args.model,
        "--conf", str(args.conf),
    ]
    if use_kafka:
        pipeline_cmd.append("--kafka")
    if not args.no_save:
        pipeline_cmd.append("--save")
    global pipeline_proc
    pipeline_proc = spawn(pipeline_cmd)

    # ── Step 5: Streamlit UI ────────────────────────────────
    print("🖥️  Starting Streamlit UI (port 8502)...")
    spawn([py, "-m", "streamlit", "run", "ui/app.py",
           "--server.headless", "true", "--server.port", "8502"],
          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)

    # ── Step 6: Open browser ────────────────────────────────
    print("\n🌐 Opening dashboard: http://localhost:8502")
    webbrowser.open("http://localhost:8502")

    print("\n" + "=" * 55)
    print("  ✅ System is running!")
    print("  Video:      OpenCV window")
    print("  Dashboard:  http://localhost:8502")
    print("  API:        http://localhost:8000/state")
    print("  Press CTRL+C or ESC (in video) to stop")
    print("=" * 55)

    # ── Keep alive — wait for pipeline then auto-shutdown ───
    try:
        pipeline_proc.wait()
        print("\n🏁 Pipeline process exited. Waiting for file writes...")
        time.sleep(2)  # give OS time to flush analytics.json to disk
    except (KeyboardInterrupt, OSError):
        pass

    # If pipeline is still running (KeyboardInterrupt), wait for it properly
    if pipeline_proc.poll() is None:
        print("   Waiting for pipeline to finish writing analytics (max 30s)...")
        try:
            pipeline_proc.wait(timeout=30)
            print("   Pipeline finished.")
        except subprocess.TimeoutExpired:
            print("   Pipeline did not exit in time. Force killing...")
            kill_proc_tree(pipeline_proc)
        time.sleep(1)

    # Now shutdown everything else
    shutdown()

    if use_kafka and not args.no_docker:
        print("\n🐳 Docker services still running.")
        print(f"   To stop: docker compose -f \"{DOCKER_COMPOSE}\" down")

    sys.exit(0)


if __name__ == "__main__":
    main()

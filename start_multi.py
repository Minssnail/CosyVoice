"""
CosyVoice Multi-Instance Auto Launcher
- Auto-detect GPU VRAM and running instances
- Calculate max safe instance count
- Launch instances on consecutive ports
"""
import subprocess
import sys
import time
import json
import os
import signal

MODEL_DIR = "pretrained_models/CosyVoice2-0.5B"
BASE_PORT = 9880
MAX_PORT = 9899  # scan range
VRAM_PER_INSTANCE_MB = 3200  # ~3.1GB per FP16 instance
VRAM_RESERVE_MB = 3000       # reserve for FFmpeg + system
RAM_PER_INSTANCE_MB = 3000   # ~3GB system RAM per instance
RAM_RESERVE_MB = 6000        # reserve 6GB for OS + p2v + FFmpeg
MIN_INSTANCES = 1
MAX_INSTANCES = 6            # 32GB RAM is the bottleneck, not 48GB VRAM

def get_gpu_info():
    """Query nvidia-smi for VRAM info"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        line = result.stdout.strip().split('\n')[0]
        parts = [p.strip() for p in line.split(',')]
        return {
            "total_mb": int(parts[0]),
            "used_mb": int(parts[1]),
            "free_mb": int(parts[2]),
            "name": parts[3]
        }
    except Exception as e:
        print(f"[ERROR] nvidia-smi failed: {e}")
        return None

def get_ram_info():
    """Query system RAM via wmic (no dependencies)"""
    try:
        # Total RAM
        r1 = subprocess.run(["wmic", "ComputerSystem", "get", "TotalPhysicalMemory", "/value"],
                            capture_output=True, text=True, timeout=10)
        total_bytes = int([l for l in r1.stdout.strip().split('\n') if 'TotalPhysicalMemory' in l][0].split('=')[1])
        # Free RAM
        r2 = subprocess.run(["wmic", "OS", "get", "FreePhysicalMemory", "/value"],
                            capture_output=True, text=True, timeout=10)
        free_kb = int([l for l in r2.stdout.strip().split('\n') if 'FreePhysicalMemory' in l][0].split('=')[1])
        total_mb = int(total_bytes / 1024 / 1024)
        free_mb = int(free_kb / 1024)
        return {"total_mb": total_mb, "used_mb": total_mb - free_mb, "free_mb": free_mb}
    except Exception:
        return {"total_mb": 32768, "used_mb": 12000, "free_mb": 20000}

def check_port_alive(port):
    """Check if a CosyVoice instance is running on given port"""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect(("127.0.0.1", port))
        s.close()
        return True
    except:
        return False

def find_running_instances():
    """Scan port range for already running instances"""
    running = []
    for port in range(BASE_PORT, MAX_PORT + 1):
        if check_port_alive(port):
            running.append(port)
    return running

def calculate_instance_count(gpu_info, ram_info, already_running):
    """Calculate how many NEW instances to launch (dual constraint: VRAM + RAM)"""
    # VRAM constraint
    vram_avail = gpu_info["free_mb"] - VRAM_RESERVE_MB
    max_by_vram = int(vram_avail / VRAM_PER_INSTANCE_MB) if vram_avail > 0 else 0

    # RAM constraint (the actual bottleneck on 32GB machines)
    ram_avail = ram_info["free_mb"] - RAM_RESERVE_MB
    max_by_ram = int(ram_avail / RAM_PER_INSTANCE_MB) if ram_avail > 0 else 0

    max_new = min(max_by_vram, max_by_ram)
    total_target = len(already_running) + max_new
    total_target = min(total_target, MAX_INSTANCES)
    new_count = total_target - len(already_running)

    print(f"  VRAM allows: {max_by_vram} new | RAM allows: {max_by_ram} new | Cap: {MAX_INSTANCES}")
    return max(new_count, 0)

def find_free_ports(count, already_running):
    """Find consecutive free ports starting from BASE_PORT"""
    free = []
    for port in range(BASE_PORT, MAX_PORT + 1):
        if port not in already_running and not check_port_alive(port):
            free.append(port)
            if len(free) >= count:
                break
    return free

def main():
    print("=" * 50)
    print("  CosyVoice Auto-Scale Launcher")
    print("=" * 50)
    
    # 1. GPU info
    gpu = get_gpu_info()
    if not gpu:
        print("[ERROR] Cannot read GPU info, exiting.")
        return
    
    print(f"\n[GPU] {gpu['name']}")
    print(f"  Total VRAM:  {gpu['total_mb']} MB")
    print(f"  Used VRAM:   {gpu['used_mb']} MB")
    print(f"  Free VRAM:   {gpu['free_mb']} MB")
    
    # 1b. RAM info
    ram = get_ram_info()
    print(f"\n[RAM]")
    print(f"  Total RAM:   {ram['total_mb']} MB")
    print(f"  Used RAM:    {ram['used_mb']} MB")
    print(f"  Free RAM:    {ram['free_mb']} MB")

    print(f"\n[CONFIG]")
    print(f"  Per instance: ~{VRAM_PER_INSTANCE_MB} MB VRAM + ~{RAM_PER_INSTANCE_MB} MB RAM")
    print(f"  Max instances: {MAX_INSTANCES}")
    
    # 2. Check existing instances
    running = find_running_instances()
    if running:
        print(f"\n[SCAN] Found {len(running)} running instance(s): ports {running}")
    else:
        print(f"\n[SCAN] No running instances found")
    
    # 3. Calculate (dual constraint: VRAM + RAM)
    new_count = calculate_instance_count(gpu, ram, running)
    if new_count == 0 and not running:
        new_count = MIN_INSTANCES
        print(f"[WARN] Low resources, starting minimum {MIN_INSTANCES} instance(s)")
    
    total = len(running) + new_count
    print(f"\n[PLAN] Launching {new_count} new instance(s), total will be {total}")
    print(f"  Est. VRAM: {total * VRAM_PER_INSTANCE_MB / 1024:.1f} GB / {gpu['total_mb'] / 1024:.1f} GB")
    print(f"  Est. RAM:  {total * RAM_PER_INSTANCE_MB / 1024:.1f} GB / {ram['total_mb'] / 1024:.1f} GB")
    
    if new_count == 0:
        print("[OK] No new instances needed. All ports are active.")
        input("\nPress Enter to exit...")
        return
    
    # 4. Find ports and launch
    ports = find_free_ports(new_count, running)
    if len(ports) < new_count:
        print(f"[WARN] Only {len(ports)} free ports available")
    
    processes = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    for i, port in enumerate(ports):
        print(f"\n[{i+1}/{len(ports)}] Launching on port {port}...")
        cmd = [
            sys.executable, "api_server.py",
            "--model_dir", MODEL_DIR,
            "--port", str(port)
        ]
        p = subprocess.Popen(cmd, cwd=script_dir)
        processes.append((port, p))
        
        if i < len(ports) - 1:
            print(f"  Waiting 15s for model to load...")
            time.sleep(15)
    
    # 5. Wait for all to be ready
    print(f"\n[WAIT] Checking all instances are ready...")
    all_ports = running + ports
    for attempt in range(30):  # max 30s wait
        alive = [p for p in all_ports if check_port_alive(p)]
        if len(alive) == len(all_ports):
            break
        time.sleep(1)
    
    alive = [p for p in all_ports if check_port_alive(p)]
    print(f"\n{'=' * 50}")
    print(f"  READY: {len(alive)} instance(s) on ports {alive}")
    print(f"  VRAM usage: ~{len(alive) * VRAM_PER_INSTANCE_MB / 1024:.1f} GB")
    print(f"  p2v engine will auto-detect these instances")
    print(f"{'=' * 50}")
    
    # 6. Keep alive
    print(f"\nPress Ctrl+C to stop all instances...")
    try:
        while True:
            time.sleep(5)
            # check if any process died
            for port, p in processes:
                if p.poll() is not None:
                    print(f"[WARN] Instance on port {port} exited with code {p.returncode}")
    except KeyboardInterrupt:
        print("\n[STOP] Shutting down all instances...")
        for port, p in processes:
            p.terminate()
        print("[DONE] All instances stopped.")

if __name__ == "__main__":
    main()


import os
import time


def main():
    filename = r"C:\Program Files (x86)\Steam\steamapps\common\Street Fighter 6\reframework\data\sf6_stream.txt"
    
    print(f"Tracking stream file in game directory: {os.path.abspath(filename)}")
    print("Waiting for Street Fighter 6 to launch and create the file...")
    
    # Wait until the game starts up and creates the file
    while not os.path.exists(filename):
        time.sleep(0.5)
        
    print("SF6 stream detected! Tail-reading 60Hz telemetry...")
    
    with open(filename, "r", errors="ignore") as f:
        # Move the reader cursor directly to the end of the file so we only read new data
        f.seek(0, os.SEEK_END)
        
        while True:
            line = f.readline()
            
            # If there is no new data yet, rest for 1ms to keep CPU usage at 0%
            if not line:
                time.sleep(0.001)
                continue
                
            # Clean up the line break and process your game data
            clean_line = line.strip()
            if clean_line:
                # Handle your frame data logic here
                print(f"Frame Data: {clean_line}")

if __name__ == "__main__":
    main()
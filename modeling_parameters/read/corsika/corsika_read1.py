from __future__ import annotations
import struct
import math
import sys

def read_float32_block(f, count):
    return list(struct.unpack(f"{count}f", f.read(count * 4)))

def read_int32(f):
    return struct.unpack("i", f.read(4))[0]

def process_file(filename):
    Rmin = 0.0
    Rstep = 0.2
    neas = 0

    dNdR_e = [0.0] * 100
    dNdR_mu = [0.0] * 100

    process_event = False 

    with open(filename, "rb") as f:
        while True:
            try:
                temp1 = read_int32(f)
                if temp1 != 22932:
                    break

                block = read_float32_block(f, 5733)
                temp2 = read_int32(f)
                if temp2 != 22932:
                    break

                for j in range(21):
                    subblock = block[j * 273:(j + 1) * 273]
                    header = struct.pack("f", subblock[0]).decode("latin1")

                    if header.startswith("EVTH"):
                        TETA_deg = subblock[10] * 180 / math.pi 
                        
                        process_event = TETA_deg >= 40
                        if process_event:
                            neas += 1

                    elif header.startswith("RUNE"):
                        raise StopIteration

                    elif not any(header.startswith(tag) for tag in ["RUNH", "EVTH", "EVTE"]):
                        if not process_event:
                            continue  

                        for k in range(39):
                            part_descr = subblock[7 * k]
                            if part_descr == 0:
                                continue

                            pid = int(part_descr / 1000)
                            x = subblock[7 * k + 4]
                            y = subblock[7 * k + 5]

                            x1 = x / 100.0
                            y1 = y / 100.0
                            r1 = math.sqrt(x1 ** 2 + y1 ** 2)

                            if r1 == 0.0:
                                continue
                            indexR = int(math.floor((math.log10(r1) - Rmin) / Rstep))
                            if 0 <= indexR < 100:
                                if pid in (2, 3):
                                    dNdR_e[indexR] += 1
                                elif pid in (5, 6):
                                    dNdR_mu[indexR] += 1

            except StopIteration:
                break
            except Exception as e:
                print("Error:", e)
                break

    with open("rho_e", "w") as f:
        for i in range(100):
            Sring = math.pi * (10 ** ((i + 1) * Rstep * 2) - 10 ** (i * Rstep * 2))
            Ro = dNdR_e[i] / Sring / neas if neas > 0 else 0
            f.write(f"{Rmin + i * Rstep + Rstep / 2:.5f}\t{Ro:.5e}\t{Sring:.5e}\n")

    with open("rho_m.txt", "w") as f:
        for i in range(100):
            Sring = math.pi * (10 ** ((i + 1) * Rstep * 2) - 10 ** (i * Rstep * 2))
            Ro = dNdR_mu[i] / Sring / neas if neas > 0 else 0
            f.write(f"{Rmin + i * Rstep + Rstep / 2:.5f}\t{Ro:.5e}\t{Sring:.5e}\n")


if __name__ == "__main__":
        process_file(sys.argv[1])

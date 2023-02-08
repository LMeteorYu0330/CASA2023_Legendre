import numpy as np
import struct


def makeVol(res, ndarray, filename):
    # res = 64
    # test = np.random.random((res, res, res, 1)).astype(np.float32)

    f = open(filename, "w")
    # ASCII Bytes 'V', 'O', 'L'
    f.write("VOL")
    f.close()

    f = open(filename, "ab")
    # File format version number
    f.write((3).to_bytes(1, byteorder="little"))

    # Encoding identifier
    f.write((1).to_bytes(4, byteorder="little"))

    # Number of cells along the X, Y, Z axis
    f.write((res).to_bytes(4, byteorder="little"))
    f.write((res).to_bytes(4, byteorder="little"))
    f.write((res).to_bytes(4, byteorder="little"))

    # Number of channels
    f.write((1).to_bytes(4, byteorder="little"))

    # Axis-aligned bounding box of the data stored in single precision
    f.write(struct.pack('<f', -0.5))
    f.write(struct.pack('<f', -0.5))
    f.write(struct.pack('<f', -0.2))
    f.write(struct.pack('<f', 1.0))
    f.write(struct.pack('<f', 1.0))
    f.write(struct.pack('<f', 1.0))

    # Volume data
    f.write(ndarray.tobytes())

    f.close()

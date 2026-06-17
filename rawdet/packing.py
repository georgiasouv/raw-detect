"""RAW preprocessing primitives: black/white normalization and RGGB packing.

These run inside the data pipeline (LoadPackedRAW), before anything learnable.
Pure numpy so they're trivially testable and framework-agnostic.
"""
import numpy as np


def normalize_raw(bayer, black_level, white_level):
    """Map sensor counts to [0, 1] using the sensor's black and white levels."""
    denom = max(float(white_level) - float(black_level), 1e-6)
    out = (bayer.astype(np.float32) - float(black_level)) / denom
    return np.clip(out, 0.0, 1.0)


def pack_rggb(bayer):
    """Pack an H x W Bayer plane into 4 x H/2 x W/2 by CFA position.

        R  G1        ->   channel 0 = R   (even rows, even cols)
        G2 B              channel 1 = G1  (even rows, odd  cols)
                          channel 2 = G2  (odd  rows, even cols)
                          channel 3 = B   (odd  rows, odd  cols)
    """
    H, W = bayer.shape
    H2, W2 = H - (H % 2), W - (W % 2)          # ensure even dims
    b = bayer[:H2, :W2]
    r  = b[0::2, 0::2]
    g1 = b[0::2, 1::2]
    g2 = b[1::2, 0::2]
    bl = b[1::2, 1::2]
    return np.stack([r, g1, g2, bl], axis=0)   # [4, H/2, W/2]


if __name__ == "__main__":
    # 4x4 ramp 0..15; check packing picks the right CFA positions and shape.
    bayer = np.arange(16, dtype=np.float32).reshape(4, 4)
    packed = pack_rggb(bayer)
    assert packed.shape == (4, 2, 2), packed.shape
    assert packed[0, 0, 0] == 0 and packed[1, 0, 0] == 1     # R=[0,0], G1=[0,1]
    assert packed[2, 0, 0] == 4 and packed[3, 0, 0] == 5     # G2=[1,0], B=[1,1]

    norm = normalize_raw(np.array([[64, 1023]], dtype=np.float32), black_level=64, white_level=1023)
    assert abs(norm[0, 0] - 0.0) < 1e-6 and abs(norm[0, 1] - 1.0) < 1e-6
    print("PACKING OK  | packed", packed.shape, "| norm range", float(norm.min()), float(norm.max()))

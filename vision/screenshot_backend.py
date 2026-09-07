"""Screenshot capture backend for A.S.T.A.

The Windows path uses Win32 GDI directly, so screenshot capture does not
require Pillow, mss, or another third-party imaging package.
"""

from __future__ import annotations

import os
import platform
import struct
import time
import zlib
from pathlib import Path


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def _encode_rgb_png(width: int, height: int, bgra: bytes) -> bytes:
    """Encode a packed BGRA byte buffer as a simple RGB PNG."""
    expected = width * height * 4
    if len(bgra) != expected:
        raise ValueError(f"Unexpected screenshot buffer size: {len(bgra)} != {expected}")

    rows = bytearray()
    source = memoryview(bgra)
    row_size = width * 4

    for y in range(height):
        rows.append(0)  # PNG filter type: none.
        row = source[y * row_size:(y + 1) * row_size]
        for x in range(0, row_size, 4):
            b, g, r, _a = row[x:x + 4]
            rows.extend((r, g, b))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(bytes(rows), 6))
        + _png_chunk(b"IEND", b"")
    )


def _capture_windows(output_path: Path) -> dict:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    handle = ctypes.c_void_p

    # Explicit prototypes prevent 64-bit Windows HDC/HBITMAP handles from
    # being truncated by ctypes' default c_int return type.
    user32.GetSystemMetrics.argtypes = [ctypes.c_int]
    user32.GetSystemMetrics.restype = ctypes.c_int
    user32.GetDC.argtypes = [wintypes.HWND]
    user32.GetDC.restype = handle
    user32.ReleaseDC.argtypes = [wintypes.HWND, handle]
    user32.ReleaseDC.restype = ctypes.c_int

    gdi32.CreateCompatibleDC.argtypes = [handle]
    gdi32.CreateCompatibleDC.restype = handle
    gdi32.CreateCompatibleBitmap.argtypes = [handle, ctypes.c_int, ctypes.c_int]
    gdi32.CreateCompatibleBitmap.restype = handle
    gdi32.SelectObject.argtypes = [handle, handle]
    gdi32.SelectObject.restype = handle
    gdi32.DeleteObject.argtypes = [handle]
    gdi32.DeleteObject.restype = wintypes.BOOL
    gdi32.DeleteDC.argtypes = [handle]
    gdi32.DeleteDC.restype = wintypes.BOOL
    gdi32.BitBlt.argtypes = [
        handle,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        handle,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.DWORD,
    ]
    gdi32.BitBlt.restype = wintypes.BOOL

    SM_XVIRTUALSCREEN = 76
    SM_YVIRTUALSCREEN = 77
    SM_CXVIRTUALSCREEN = 78
    SM_CYVIRTUALSCREEN = 79
    SRCCOPY = 0x00CC0020
    CAPTUREBLT = 0x40000000
    DIB_RGB_COLORS = 0
    BI_RGB = 0

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", wintypes.DWORD),
            ("biWidth", wintypes.LONG),
            ("biHeight", wintypes.LONG),
            ("biPlanes", wintypes.WORD),
            ("biBitCount", wintypes.WORD),
            ("biCompression", wintypes.DWORD),
            ("biSizeImage", wintypes.DWORD),
            ("biXPelsPerMeter", wintypes.LONG),
            ("biYPelsPerMeter", wintypes.LONG),
            ("biClrUsed", wintypes.DWORD),
            ("biClrImportant", wintypes.DWORD),
        ]

    class BITMAPINFO(ctypes.Structure):
        _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]

    gdi32.GetDIBits.argtypes = [
        handle,
        handle,
        wintypes.UINT,
        wintypes.UINT,
        ctypes.c_void_p,
        ctypes.POINTER(BITMAPINFO),
        wintypes.UINT,
    ]
    gdi32.GetDIBits.restype = ctypes.c_int

    x = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    y = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    width = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
    height = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
    if width <= 0 or height <= 0:
        raise RuntimeError("Windows returned an invalid virtual screen size.")

    desktop_dc = user32.GetDC(0)
    if not desktop_dc:
        raise RuntimeError("GetDC failed while capturing the desktop.")

    memory_dc = gdi32.CreateCompatibleDC(desktop_dc)
    bitmap = gdi32.CreateCompatibleBitmap(desktop_dc, width, height)
    if not memory_dc or not bitmap:
        if memory_dc:
            gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(0, desktop_dc)
        raise RuntimeError("Unable to allocate the Windows screenshot buffer.")

    old_bitmap = gdi32.SelectObject(memory_dc, bitmap)

    try:
        if not gdi32.BitBlt(
            memory_dc,
            0,
            0,
            width,
            height,
            desktop_dc,
            x,
            y,
            SRCCOPY | CAPTUREBLT,
        ):
            raise RuntimeError("BitBlt failed while copying the desktop image.")

        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = width
        bmi.bmiHeader.biHeight = -height  # top-down DIB
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = BI_RGB

        buffer_size = width * height * 4
        pixels = (ctypes.c_ubyte * buffer_size)()
        copied = gdi32.GetDIBits(
            memory_dc,
            bitmap,
            0,
            height,
            ctypes.cast(pixels, ctypes.c_void_p),
            ctypes.byref(bmi),
            DIB_RGB_COLORS,
        )
        if copied != height:
            raise RuntimeError(f"GetDIBits captured {copied} rows instead of {height}.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(_encode_rgb_png(width, height, bytes(pixels)))
    finally:
        gdi32.SelectObject(memory_dc, old_bitmap)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(0, desktop_dc)

    return {
        "path": str(output_path.resolve()),
        "width": width,
        "height": height,
        "platform": "windows",
    }


def _capture_with_pillow(output_path: Path) -> dict:
    try:
        from PIL import ImageGrab
    except ImportError as exc:
        raise RuntimeError(
            "Screenshot backend is unavailable on this platform. Install Pillow "
            "or provide an injected screenshot backend."
        ) from exc

    image = ImageGrab.grab(all_screens=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    return {
        "path": str(output_path.resolve()),
        "width": image.width,
        "height": image.height,
        "platform": platform.system().lower(),
    }


def capture_screenshot(output_dir: str | os.PathLike = "runtime/screenshots") -> dict:
    """Capture the full virtual desktop and return metadata about the image."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"asta_{timestamp}_{time.time_ns() % 1_000_000_000:09d}.png"

    if platform.system() == "Windows":
        return _capture_windows(output_path)

    return _capture_with_pillow(output_path)

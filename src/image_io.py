import base64
import binascii
import io
import mimetypes
import os
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


def decode_base64_to_bytes(image_base64: str) -> bytes:
    value = image_base64.strip()
    if "," in value and value.lower().startswith("data:"):
        value = value.split(",", 1)[1]

    try:
        return base64.b64decode(value, validate=True)
    except binascii.Error:
        compact = "".join(value.split())
        return base64.b64decode(compact, validate=True)


def load_url_bytes(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "OpenSDI/1.0"})
    with urlopen(req, timeout=30) as response:
        return response.read()


def load_file_bytes(path: str) -> bytes:
    file_path = Path(path).expanduser()

    if not file_path.is_file():
        raise ValueError(f"invalid_image_path: File not found: {path}")

    return file_path.read_bytes()


def guess_image_mime(image_bytes: bytes, image_ref: str | None = None) -> str:
    if image_ref:
        guessed, _ = mimetypes.guess_type(image_ref)
        if guessed and guessed.startswith("image/"):
            return guessed

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            fmt = (img.format or "").lower()
            if fmt == "jpeg":
                return "image/jpeg"
            if fmt == "png":
                return "image/png"
            if fmt == "webp":
                return "image/webp"
            if fmt == "gif":
                return "image/gif"
    except Exception:
        pass

    return "image/jpeg"


def bytes_to_data_url(image_bytes: bytes, image_ref: str | None = None) -> str:
    mime = guess_image_mime(image_bytes, image_ref)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def bytes_from_ref(image_ref: str) -> bytes:
    ref = image_ref.strip()

    if ref.startswith(("http://", "https://")):
        return load_url_bytes(ref)

    if ref.startswith("data:image/"):
        return decode_base64_to_bytes(ref)

    # Support local file path, e.g. image.png, /tmp/image.jpg
    if os.path.isfile(os.path.expanduser(ref)):
        return load_file_bytes(ref)

    return decode_base64_to_bytes(ref)


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    with Image.open(io.BytesIO(image_bytes)) as img:
        return img.convert("RGB").copy()


def image_from_ref(image_ref: str) -> Image.Image:
    return bytes_to_pil(bytes_from_ref(image_ref))


def image_ref_for_openai(image_ref: str) -> str:
    ref = image_ref.strip()

    if ref.startswith(("http://", "https://")):
        return ref

    if ref.startswith("data:image/"):
        return ref

    image_bytes = bytes_from_ref(ref)
    return bytes_to_data_url(image_bytes, ref)
import base64
import binascii
import io
from urllib.request import Request, urlopen

from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


def decode_base64_to_bytes(image_base64: str) -> bytes:
    if "," in image_base64 and image_base64.lower().startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]

    try:
        return base64.b64decode(image_base64, validate=True)
    except binascii.Error:
        compact = "".join(image_base64.split())
        return base64.b64decode(compact, validate=True)


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    with Image.open(io.BytesIO(image_bytes)) as img:
        return img.convert("RGB").copy()


def load_url_bytes(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "OpenSDI/1.0"})
    with urlopen(req, timeout=15) as response:
        return response.read()


def bytes_from_ref(image_ref: str) -> bytes:
    if image_ref.startswith("http://") or image_ref.startswith("https://"):
        return load_url_bytes(image_ref)
    return decode_base64_to_bytes(image_ref)


def image_from_ref(image_ref: str) -> Image.Image:
    return bytes_to_pil(bytes_from_ref(image_ref))

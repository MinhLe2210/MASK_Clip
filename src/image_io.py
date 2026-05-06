import base64
import binascii
import io
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


def bytes_from_ref(image_ref: str) -> bytes:
    if image_ref.startswith(("http://", "https://")):
        return load_url_bytes(image_ref)
    return decode_base64_to_bytes(image_ref)


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    with Image.open(io.BytesIO(image_bytes)) as img:
        return img.convert("RGB").copy()


def image_from_ref(image_ref: str) -> Image.Image:
    return bytes_to_pil(bytes_from_ref(image_ref))


def image_ref_for_openai(image_ref: str) -> str:
    if image_ref.startswith(("http://", "https://")):
        return image_ref
    if image_ref.startswith("data:image/"):
        return image_ref
    return f"data:image/jpeg;base64,{image_ref}"

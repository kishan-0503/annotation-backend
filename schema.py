from pydantic import BaseModel


class Coordinate(BaseModel):
    image_url: str
    x: int
    y: int
    w: int
    h: int


class TrainingCoordinate(BaseModel):
    image: bytes
    x: int
    y: int
    w: int
    h: int

class OCRRequest(BaseModel):
    x: int
    y: int
    w: int
    h: int
    base64_image: str
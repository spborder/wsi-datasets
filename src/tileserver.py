"""

Simple tile server implementation for WSIs using FastAPI

"""

import large_image
import uvicorn
from fastapi import APIRouter, FastAPI, Request, Response


class TileServer:
    def __init__(self, slides: list, host: str = "0.0.0.0", port: int = 8000):
        self.slides = slides
        self.host = host
        self.port = port

        self.app = FastAPI(title="WSI Tile Server")

        self.router = APIRouter()
        self.router.add_api_route("/", self.root, methods=["GET", "OPTIONS"])
        self.router.add_api_route(
            "/slides", self.list_slides, methods=["GET", "OPTIONS"]
        )
        self.router.add_api_route(
            "/{slide_id}/tiles/{z}/{x}/{y}.png",
            self.get_tile,
            methods=["GET", "OPTIONS"],
        )

    def root(self):
        return {"message": "WSI Tile Server"}

    def list_slides(self):
        return {
            "slides": [
                {"slide.id": slide._id, "slide.filepath": slide.filepath}
                for slide in self.slides
            ]
        }

    async def get_tile(self, slide_id: str, z: int, x: int, y: int):

        if any([i < 0 for i in [z, x, y]]):
            # Don't return anything, but successfully
            return Response(status_code=200)

        if slide_id not in [slide._id for slide in self.slides]:
            return Response(status_code=404)

        else:
            slide_idx = [slide._id for slide in self.slides].index(slide_id)
            slide = self.slides[slide_idx]

            try:
                raw_tile = slide.tile_source.getTile(x=x, y=y, z=z)
                return Response(
                    content=raw_tile, media_type="image/png", status_code=200
                )

            except large_image.exceptions.TileSourceXYZRangeError:
                return Response(status_code=200)

    def start(self):

        self.app.include_router(self.router)
        uvicorn.run(self.app, host=self.host, port=self.port)

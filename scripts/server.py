import requests

import io
import json

import click
import torch
import numpy as np
import uvicorn
import clip #ensure you are installing the CLIP.git not the clip package
import re

from fastapi import FastAPI, File, Form
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Sequence, Callable
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
from typing_extensions import Annotated
from threading import Lock
from io import BytesIO



class Point(BaseModel):
    x: int
    y: int


class Points(BaseModel):
    points: Sequence[Point]
    points_labels: Sequence[int]


class Box(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class TextPrompt(BaseModel):
    text: str


def segment_image(image_array: np.ndarray, segmentation_mask: np.ndarray):
    segmented_image = np.zeros_like(image_array)
    segmented_image[segmentation_mask] = image_array[segmentation_mask]
    return segmented_image


def retrieve(
    elements: Sequence[np.ndarray],
    search_text: str,
    preprocess: Callable[[Image.Image], torch.Tensor],
    model, device=torch.device('cpu')
) -> torch.Tensor:
    with torch.no_grad():
        preprocessed_images = [preprocess(Image.fromarray(
            image)).to(device) for image in elements]
        tokenized_text = clip.tokenize([search_text]).to(device)
        stacked_images = torch.stack(preprocessed_images)
        image_features = model.encode_image(stacked_images)
        text_features = model.encode_text(tokenized_text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        probs = (100.0 * image_features @ text_features.T)
    return probs[:, 0].softmax(dim=-1)

# Here we can change the used model, so as we create our own or if we wish to switch to one of the other three models,
# this is where we can do it.
@click.command()
@click.option('--model',
              default='vit_h',
              help='model name',
              type=click.Choice(['vit_b', 'vit_l', 'vit_h']))
@click.option('--model_path', default='model/sam_vit_h_4b8939.pth', help='model path')
@click.option('--port', default=8000, help='port')
@click.option('--host', default='0.0.0.0', help='host')
def main(
        model="vit_h",
        model_path="model/sam_vit_h_4b8939.pth",
        port=8000,
        host="0.0.0.0",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    build_sam = sam_model_registry[model]
    model = build_sam(checkpoint=model_path).to(device)
    predictor = SamPredictor(model)
    mask_generator = SamAutomaticMaskGenerator(model)
    model_lock = Lock()

    clip_model, preprocess = clip.load("ViT-B/16", device=device)

    app = FastAPI()

    @app.get('/')
    def index():
        return {"code": 0, "data": "Hello World"}

    def compress_mask(mask: np.ndarray):
        # with open('some_File_timestamp.raw', )
        flat_mask = mask.ravel()
        idx = np.flatnonzero(np.diff(flat_mask))
        idx = np.concatenate(([0], idx + 1, [len(flat_mask)]))
        counts = np.diff(idx)
        values = flat_mask[idx[:-1]]
        compressed = ''.join(
            [f"{c}{'T' if v else 'F'}" for c, v in zip(counts, values)])
        return compressed

    def generate_overlay(compressed, imgx, imgy, filename):
        counts = []
        values = []
        splits = re.split('(T|F)', compressed)
        # print(splits)
        pixel_color = []

        for each in splits:
            if each == 'T':
                values.append('T')
            elif each == 'F':
                values.append('F')
            elif each != 'T' and each != 'F' and each != '':
                counts.append(int(each))

        i = 0
        for each in counts:
            for pixel in range(each):
                if (i % 2 == 0):
                    pixel_color.append(False)  # black
                else:
                    pixel_color.append(True)  # white
            i += 1

        pixel_color = np.array(pixel_color)

        overlay = Image.fromarray(pixel_color.reshape((imgy, imgx)).astype('uint8') * 255)
        overlay_stream = overlay.tobytes() # creates the encoded bytestream which can be rebuilt on the php side

        return overlay_stream

    # Gets a UUID from the dropdown on the frontend, then checks for that UUID on the Template Site DB, returns a byte string for the image, it gets reconstructed and displayed on the front
    @app.post('/api/upload')
    async def api_open(
            uuid: Annotated[str, Form(...)],
            file: Annotated[bytes, File()], # not sure how this might work yet, but if we need to pull the image from somewhere else, we can push the bytes of the file and reconstruct on the frontend, reverse of how it's done right now
    ):
        return {"code": 0, "data": [uuid, file]} # returns the filename of the file we want to upload

    # Gets the UUIDs in the php site and sends it to the react frontend to allow choosing predefined files
    @app.post('/api/populate')
    async def api_populate(
        filenames: Annotated[str, Form(...)]
    ):
        available_files = json.loads(filenames)

        return {"code": 0, "data": available_files["filenames"]} # returns a list of filenames which function as UUIDs corresponding to available checkpoints on the php side

    @app.post('/api/download')
    async def api_download(
            file: Annotated[bytes, File()],
            overlay_filename: Annotated[str, Form(...)],
            imgx: Annotated[str, Form(...)],
            imgy: Annotated[str, Form(...)],
            points_filename: Annotated[str, Form(...)],
            points: Annotated[str, Form(...)],
    ):
        ps = Points.parse_raw(points)
        input_points = np.array([[p.x, p.y] for p in ps.points])
        input_labels = np.array(ps.points_labels)
        image_data = Image.open(io.BytesIO(file))
        image_data = np.array(image_data)
        with model_lock:
            predictor.set_image(image_data)
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )
            predictor.reset_image()

        x_dim = re.split('{|:|}|"', imgx)
        y_dim = re.split('{|:|}|"', imgy)

        of_dict = json.loads(overlay_filename)

        pf_dict = json.loads(points_filename)
        points_dict = json.loads(points)

        storage_url = "http://localhost:8090/save_image" #maybe??

        overlay_data = generate_overlay(compress_mask(np.array(masks[2])), int(x_dim[5]), int(y_dim[5]), of_dict['filename'])

        r = requests.post(url=storage_url, params={"image_data":file, "image_x":imgx, "image_y":imgy,
                                                   "overlay_filename":overlay_filename, "overlay_data":overlay_data,
                                                   "points_filename":points_filename, "points":points}) #i think that these need processing otherwise it's like double JSONing

        # Deprecated code which generates and saves the image and file here on the python server, but we wanna push the files
        # over to the php side instead.

        '''url = generate_overlay(compress_mask(np.array(masks[2])), int(x_dim[5]), int(y_dim[5]), of_dict['filename'])

        with open('filenames.txt', 'w', encoding='utf-8') as f:
            f.write(overlay_filename)
            f.write(points_filename)


        with open('dataset/' + pf_dict['filename'] + '.json', 'w', encoding='utf-8') as f:
            json.dump(points_dict, f)'''

        return {"code": 0, "data": r.json()} # r.json() is the response we get from requests.post(), so this can give us nice error messages and whatever else

    #Inserts points sent here in the form of a valid JSON object which is produced by the frontend
    @app.post('/api/copy-paste')
    async def api_insert(
            file: Annotated[bytes, File()],
            points: Annotated[str, Form(...)],
    ):
        ps = Points.parse_raw(points)
        input_points = np.array([[p.x, p.y] for p in ps.points])
        input_labels = np.array(ps.points_labels)
        image_data = Image.open(io.BytesIO(file))
        image_data = np.array(image_data)
        with model_lock:
            predictor.set_image(image_data)
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )
            predictor.reset_image()
        masks = [
            {
                "segmentation": compress_mask(np.array(mask)),
                "stability_score": float(scores[idx]),
                "bbox": [0, 0, 0, 0],
                "area": np.sum(mask).item(),
            }
            for idx, mask in enumerate(masks)
        ]
        masks = sorted(masks, key=lambda x: x['stability_score'], reverse=True)
        return {"code": 0, "data": masks[:]}

    @app.post('/api/point')
    async def api_points(
            file: Annotated[bytes, File()],
            points: Annotated[str, Form(...)],
    ):
        ps = Points.parse_raw(points)
        input_points = np.array([[p.x, p.y] for p in ps.points])
        input_labels = np.array(ps.points_labels)
        image_data = Image.open(io.BytesIO(file))
        image_data = np.array(image_data)
        with model_lock:
            predictor.set_image(image_data)
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )
            predictor.reset_image()
        masks = [
            {
                "segmentation": compress_mask(np.array(mask)),
                "stability_score": float(scores[idx]),
                "bbox": [0, 0, 0, 0],
                "area": np.sum(mask).item(),
            }
            for idx, mask in enumerate(masks)
        ]
        masks = sorted(masks, key=lambda x: x['stability_score'], reverse=True)
        return {"code": 0, "data": masks[:]}

    @app.post('/api/box')
    async def api_box(
        file: Annotated[bytes, File()],
        box: Annotated[str, Form(...)],
    ):
        b = Box.parse_raw(box)
        input_box = np.array([b.x1, b.y1, b.x2, b.y2])
        image_data = Image.open(io.BytesIO(file))
        image_data = np.array(image_data)
        with model_lock:
            predictor.set_image(image_data)
            masks, scores, logits = predictor.predict(
                box=input_box,
                multimask_output=False,
            )
            predictor.reset_image()
        masks = [
            {
                "segmentation": compress_mask(np.array(mask)),
                "stability_score": float(scores[idx]),
                "bbox": [0, 0, 0, 0],
                "area": np.sum(mask).item(),
            }
            for idx, mask in enumerate(masks)
        ]
        masks = sorted(masks, key=lambda x: x['stability_score'], reverse=True)
        return {"code": 0, "data": masks[:]}

    @app.post('/api/everything')
    async def api_everything(file: Annotated[bytes, File()]):
        image_data = Image.open(io.BytesIO(file))
        image_array = np.array(image_data)
        masks = mask_generator.generate(image_array)
        arg_idx = np.argsort([mask['stability_score']
                              for mask in masks])[::-1].tolist()
        masks = [masks[i] for i in arg_idx]
        for mask in masks:
            mask['segmentation'] = compress_mask(mask['segmentation'])
        return {"code": 0, "data": masks[:]}

    @app.post('/api/clip')
    async def api_clip(
            file: Annotated[bytes, File()],
            prompt: Annotated[str, Form(...)],
    ):
        text_prompt = TextPrompt.parse_raw(prompt)
        image_data = Image.open(io.BytesIO(file))
        image_array = np.array(image_data)
        masks = mask_generator.generate(image_array)
        cropped_boxes = []
        for mask in masks:
            bobx = [int(x) for x in mask['bbox']]
            cropped_boxes.append(segment_image(image_array, mask["segmentation"])[
                bobx[1]:bobx[1] + bobx[3], bobx[0]:bobx[0] + bobx[2]])
        scores = retrieve(cropped_boxes, text_prompt.text,
                          model=clip_model, preprocess=preprocess, device=device)
        top = scores.topk(5)
        masks = [masks[i] for i in top.indices]
        for mask in masks:
            mask['segmentation'] = compress_mask(mask['segmentation'])
        return {"code": 0, "data": masks[:]}

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

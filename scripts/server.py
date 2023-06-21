import io
import os
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
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry, build_sam
from PIL import Image
from typing_extensions import Annotated
from threading import Lock
from io import BytesIO

<<<<<<< Updated upstream
=======
import os
os.path.join
import build_sam


>>>>>>> Stashed changes
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

def setModel(modelname, modelfile):
    return modelname, modelfile

model_file = "sam_vit_b_01ec64.pth" #default
model_name = "vit_b" #default

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

sam = sam_model_registry[model_name](checkpoint="model/"+model_file)
sam.to(device=device)
predictor = SamPredictor(sam)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mask_generator = SamAutomaticMaskGenerator(sam)
model_lock = Lock()

model_choices = ['vit_b', 'vit_h', 'vit_l'] #these are the supported model names in build_sam
model_files = ['sam_vit_b_01ec64.pth', 'sam_vit_h_4b8939.pth', 'sam_vit_l_0b3195.pth'] #these are the file paths which we have downloaded


def rebuildSAM():
    sam = sam_model_registry[model_name](checkpoint="model/"+model_file)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    mask_generator = SamAutomaticMaskGenerator(sam)
    model_lock = Lock()

    return predictor, mask_generator, model_lock

# Here we can change the used model, so as we create our own or if we wish to switch to one of the other three models,
# this is where we can do it.
@click.command()
@click.option('--model',
              default=model_name,
              help='model name',
              type=click.Choice(model_choices)) #need to add more options here as we add new models
@click.option('--model_path', default='model/'+model_file, help='model path')
@click.option('--port', default=8000, help='port')
@click.option('--host', default='0.0.0.0', help='host')
def main(
        model=model_name, # used to just say "vit_b"
        model_path="model/"+model_file,
        port=8000,
        host="0.0.0.0",
):
    #global model_name, model_file, predictor, mask_generator, model_lock

    #predictor, mask_generator, model_lock = rebuildSAM()

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
        overlay.save('overlays/' + filename + '.png', 'PNG')

        url = 'localhost:8000/'+filename+'.png'
        return url

    @app.post('/api/populate')
    async def api_populate(
    ):
        return {"code": 0, "data": model_choices}

    @app.post('/api/select-model')
    async def api_select(
        file: Annotated[bytes, File()],
        redo: Annotated[bool, Form(...)],
        points: Annotated[str, Form(...)],
        modelname = Annotated[str, Form(...)],
    ):
        global model_name, model_file, predictor, mask_generator, model_lock # Calls back to old vars instead of making local refs
        # Add new cases for new models which can be selected
        match modelname:
            case 'vit_b':
                model_name, model_file = setModel(modelname, model_files[0])
            case 'vit_h':
                model_name, model_file = setModel(modelname, model_files[1])
            case 'vit_l':
                model_name, model_file = setModel(modelname, model_files[2])

        predictor, mask_generator, model_lock = rebuildSAM()

        if redo: #only runs if redo is true, meaning that there are points to be redone with the new model selection
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
            return {"code": 2, "data": masks[:], "model": model_name}
        else:
            return {"code": 0, "model": model_name}

    @app.post('/api/download')
    async def api_download(
            file: Annotated[bytes, File()],
            filename: Annotated[str, Form(...)],
            imgx: Annotated[str, Form(...)],
            imgy: Annotated[str, Form(...)],
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
        file_name = re.split('{|:|}|"', filename)

        url = generate_overlay(compress_mask(np.array(masks[2])), int(x_dim[5]), int(y_dim[5]), file_name[5])
        return {"code": 0, "data": url}

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

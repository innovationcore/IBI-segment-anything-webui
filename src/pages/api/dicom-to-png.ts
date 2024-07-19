// Next.js API route support: https://nextjs.org/docs/api-routes/introduction
import type { NextApiRequest, NextApiResponse } from 'next';
import formidable from 'formidable';
import { promises as fs } from 'fs';
import dicomParser from 'dicom-parser';
import { PNG } from 'pngjs/browser';

export const config = {
  api: {
    bodyParser: false,
  },
};

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  try {
    // Create a temporary directory for uploads
    await fs.mkdir('./tmp/', { recursive: true });

    // Configure formidable to save uploaded files to the temporary directory
    const form = formidable({ uploadDir: './tmp/' });

    // Parse the form to extract the uploaded file
    const { fields, files } = await new Promise<{ fields: formidable.Fields; files: formidable.Files }>((resolve, reject) => {
      form.parse(req, (err, fields, files) => {
        if (err) {
          reject(err);
          return;
        }
        resolve({ fields, files });
      });
    });

    // Get the uploaded file path
    const file_list = files['file'] as formidable.File[];
    const filepath = file_list[0]['filepath'];

    // Read the DICOM file
    const dicomFileBuffer = await fs.readFile(filepath);

    // Parse the DICOM file
    const dataSet = dicomParser.parseDicom(dicomFileBuffer);

    // Extract pixel data from the DICOM file
    const pixelDataElement = dataSet.elements.x7fe00010;
    const pixelData = new Uint8Array(dataSet.byteArray.buffer, pixelDataElement.dataOffset, pixelDataElement.length);

    // Get image dimensions
    const rows = dataSet.uint16('x00280010');
    const columns = dataSet.uint16('x00280011');

    // Create a new PNG image
    const png = new PNG({
      width: columns,
      height: rows,
      colorType: 0, // Grayscale
      inputHasAlpha: false,
    });

    if (rows !== undefined && columns !== undefined) {
      // Copy the pixel data to the PNG image
      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < columns; x++) {
          const pixelValue = pixelData[y * columns + x];
          const index = (y * columns + x) * 4;
          png.data[index] = pixelValue;
          png.data[index + 1] = pixelValue;
          png.data[index + 2] = pixelValue;
          png.data[index + 3] = 255; // Alpha channel
        }
      }
    }

    // Convert PNG to buffer
    const pngBuffer = PNG.sync.write(png);

    // Set response headers and send PNG buffer
    res.setHeader('Content-Type', 'image/png');
    res.status(200).send(pngBuffer);

    // Clean up the temporary file
    await fs.unlink(filepath);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
}

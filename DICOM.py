import cv2
import os
import pydicom
import numpy as np

inputdir = 'Exploding Rats/Rat 1/'
outdir = 'Exploding Rats/Rat 1/'

# Ensure output directory exists
os.makedirs(outdir, exist_ok=True)

# Get list of DICOM files
test_list = [f for f in os.listdir(inputdir) if f.lower().endswith('.dcm')]

if not test_list:
    print("No DICOM files found in the input directory.")
else:
    for f in test_list[:10]:  # remove "[:10]" to process all images
        try:
            ds = pydicom.read_file(os.path.join(inputdir, f))  # read dicom image
            img = ds.pixel_array  # get image array

            # Print type and shape for debugging
            print(f"Processing {f}: Type = {type(img)}, Shape = {img.shape}, dtype = {img.dtype}")

            # Handle 3D image
            if len(img.shape) == 3:
                # Use Maximum Intensity Projection (MIP)
                img = np.max(img, axis=0)

            # Ensure image is grayscale (2D)
            if len(img.shape) == 2:
                # Normalize image to 8-bit if necessary
                if img.dtype != np.uint8:
                    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                success = cv2.imwrite(os.path.join(outdir, f.replace('.dcm', '.png')), img)

                if not success:
                    print(f"Failed to save image {f.replace('.dcm', '.png')}")
            else:
                print(f"Unexpected image shape for {f}: {img.shape}. Expected a 2D grayscale image.")

        except Exception as e:
            print(f"Error processing file {f}: {e}")

    print("Conversion completed.")

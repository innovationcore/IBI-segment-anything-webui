# Web Sam for UKHC

<p>WebSAM for UKHC is a three part package:</p>

[WebSAM for UKHC Repository](https://github.com/innovationcore/WebSAM-for-UKHC)

<p>WebSAM for UKHC leverages Segment Anything to provide in-browser image segmentations.
It consists of a ReactJS frontend, and a FastAPI python backend which contains API functions for saving and loading images, generating segmentation masks, and uploading JSON files with points to replicate results.</p>

<p>To create a segmentation, all you have to do is upload a file, click, and then when done clicking you can save the image to the database. Additionally, the threshold slider allows you to tweak segmentation around the click points.</p>

## Interactive parts of this site include:
|Item|Function|
|--------|:---------|
|Click| this allows for you to place points on an image that generate a mask|
|Box| this allows you to drag a box over an area of interest to generate a mask|
|Everything| this is an "automatic" segmenter which takes the onus of placing points off the user|
|Send Query| this button and it's text box allows you to use CLIP from OpenAI to write a query to segment something from the image|
|Upload Points JSON File| this button allows you to select a .json file which contains the properly formatted points to recreate a previous segmentation (or you could make one up)|
|Clean Segment| get rid of all segmentations and points, leave the uploaded image|
|Clean All| remove everything, basically refreshes the page|
|Undo Last Point| will remove the latest point from the image and re-run the segmentation|

<p>There is a tab which only shows up when an image is uploaded and the first point has been placed. The button is Download Results which allows you to download the file, generate and save a mask file, and save a json file with the points used to create the segmentation to the WebSAM Image Database.</p>

# Sturdy Waddle
[Sturdy Waddle Repository](https://github.com/innovationcore/sturdy-waddle)

<p>Sturdy Waddle is a fork of the template site. It has been modified to have a page which features the WebSAM interface, and a page which shows the files which are stored in the database. Minimal changes have been made from the original template site.</p>

# WebSAM Image Database
[WebSAM Image Database Repository](https://github.com/innovationcore/WebSAM-Image-Database)

<p>WebSAM Image Database contains code to save images that are downloaded frm WebSAM-for-UKHC, as well as to provide identifying information and thumbnails for the images to be placed in the processed tab of sturdy-waddle.</p>

<p>This is not an interactive site, it is simply a stand-in for a database implementation down the road. For now, it creates and saves files to a folder called datasets.</p>

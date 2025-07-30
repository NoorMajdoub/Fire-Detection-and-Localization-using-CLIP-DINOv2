Notessss


## What is Dinov2
Self-DIstillation with NO labels It’s a self-supervised vision foundation model

## What tasks is it used for
class/semantic labeling/ Image segmentation (what we will used if for here )/object detection/d3 redefinritoon It's a vision transformer that generates dense patch-level embeddings, good for tasks like:

Image classification

Object detection

Semantic segmentation

Anomaly detection

Image retrieval

## What is the flow of the input to output
Input image ➡️ Augmentations (viws) : we take an image and apply a bunch of ttansformations (crop rotate ect) that way w we get differnet views of the same image

Patch embeddings computed (ViT backbone) no idea yet why who what

Self-distillation loss: student tries to match teacher's representation nope dont get it either yet

Final output: feature maps or global descriptors (we get overal feature map from the teach and patch level features maps)

How it is used in this project

detect the regions of fire in the image Howwww

## help
Extract patch-wise features from DINOv2.

Compare each patch to the global representation (e.g., from a normal scene).

Compute similarity → low similarity = anomaly → fire region.

Visualize this as a heatmap (Anomaly Map + Color Overlay).

## How DINOv2 Detected Fire Regions
we are working patches level (not pixels ) when getting the embeddingof those patches we are getting the semantic signicance also , so when we also used dino on the reference pic with no fire we got the compare the patches by cosine sim (we comparing patch per patch) for the ones with fire the diff is gonna be big , we like that managed to capture the regions aka patches with fire in em ps choice of reference pic is important ..duh

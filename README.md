# ICDEC 2024 Challenge: Vehicle Detection in Various Weather Conditions(VDVWC)

A YOLO (You Only Look Once) based model trained on the limited personal dataset given here : [dataset_link](https://github.com/Sourajit-Maity/juvdv2-vdvwc.git)

#### CLASSES : 

The dataset was quite imbalanced  , The `car` class had much more images than the other classes . Here is a basic visualization which shows the class imbalance 

![1720586899664](image/README/1720586899664.png)

![1720586938410](image/README/1720586938410.png)

This high imbalance might lead to model bias which might not work great with the unseen data 

---

### YOLO MODEL WITH CLASS IMBALANCE

First of all , lets see how does the model performs with the class imbalances present in the dataset through the confusion matrix which might be the best way right now to judge if model is performing well or not 


if the image is not visible , here;s the link for the excalidraw notebook with other comparisions as well : [Excalidraw_notebook ](https://excalidraw.com/#json=N0QYiNPK9x-QPxrEU3Izt,em0B7E7QtfhwAaLFPTn84w)

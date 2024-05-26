# Football tracking

I utilise a computer vision dataset from [Roboflow Universe](https://universe.roboflow.com/) for training. Original video footage and annotations were then converted by roboflow into a workable dataset. Here is the [link](https://github.com/roboflow/notebooks/blob/main/notebooks/how-to-track-football-players.ipynb) to the tutorial I followed.

## Stage 1 - object detection

what do I see on image and where do I see it, bounding box for location and class name for type of object

## Stage 2 - tracking objects

What we see from first frame is what they see on the next frame, a unique id is given to each bounding box

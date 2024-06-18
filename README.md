# Football tracking

I utilise a computer vision dataset from [Bundesliga Data Shootout](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/data) for training. Comprises video recordings of nine football games divided into halves.

## Stage 1 - Object Detection

I get YOLO to detect players and ball in video clips creating bounding box for location and class name for type of object
![Football Tracking Output](https://raw.githubusercontent.com/rob-sullivan/ai/football-tracking/football-tracking/output.PNG)

## Stage 2 - Tracking Objects

Go from frame to frame and track objects with a unique id given to each bounding box

## Stage 3 - Football Stats ELT

Determine stats such as % ball possession, player formations, etc.

## Stage 4 - Playback Dashboard

Create a 2d representation of a pitch and project stats onto it allowing user to playback and fastforward to key moments in match.

## Stage 5 - Sports Journalist

Pass stats to ChatGPT as context via api and allow user to query it.

# Data Directory

Place your dataset in this directory using the following structure:

```
data/
  train/
    <class_name>/*.mp4
  val/
    <class_name>/*.mp4
  test/
    <class_name>/*.mp4
```

Examples for the default binary task:
```
data/
  train/
    shoplifting/
      clip001.mp4
    no_shoplifting/
      clip002.mp4
```

You can generate this layout automatically from a CSV file containing `path,label` columns by running:
```bash
python scripts/prepare_dataset.py --csv annotations.csv --output-root data --splits 0.7 0.2 0.1
```

Videos should be short clips (a few seconds) encoded as MP4, although other formats supported by FFmpeg will also work.

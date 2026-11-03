# GE2340 lost-and-found matcher

CNN image classification with a linear classifier.

A lost query is text: location, time, date, category, and colour.
Each found record has those fields plus one photo.
The model returns the five found records with the highest match probability.

## Where to put photos

Put every found photo in `data/images/found/`.
Keep the original filename. Do not rename a file to store its label.

- A photo listed in `data/labels/photos.csv` with `labelled` = `1` is labelled.
- A photo in the folder with no row, or with `labelled` = `0`, is unlabelled.
- Unlabelled photos stay in the same folder. Do not make a second copy.

Delete `example_delete_me.jpg` from the CSV before real uploads. That row is only an example.

## How to mark a labelled photo

1. Copy the photo into `data/images/found/`.
2. Add one row to `data/labels/photos.csv`.
3. `filename` must match the file name exactly, including `.jpg` or `.png`.
4. Set `labelled` to `1`.
5. Fill `category` and `colour` using only the names in `data/labels/allowed_labels.txt`.
6. Fill location, time (`HH:MM`), and date (`YYYY-MM-DD`).

Two labels do not fit in a filename. Category and colour are written in the CSV, next to the filename.

## Queries and pairs

`data/labels/queries.csv` is a lost query. It has no photo.

`data/labels/pairs.csv` trains the linear classifier.

- `is_match` = `1`: this found photo is the correct record for that query.
- `is_match` = `0`: this found photo is not the correct record.

The correct photo must already have `labelled` = `1` in `photos.csv`.
Do not upload ID cards, student cards, or photos that show a face.

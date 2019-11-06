# chilis_is_the_new_golf_course

There are a few distinct components in this repo:

1. A jupyter notebook used to scrape officequotes.net for structured, labeled quotes from office characters.

2. a CSV file of the raw scraped data.

3. Preprocessing and model training code used to model the data and export saved models.

4. A flask app to load the various models, predict on new data and display results.

You'll find a conda yml file in the text_model directory from which you can install the necessary dependencies into a new environment to run the scripts and flask app.

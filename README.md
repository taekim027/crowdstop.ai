# crowdstop.ai

Authors: Tae Kim, Jeremy Lan, Michelle Lee

## Mission Statement
Implement an crowd monitoring system using a network of security cameras to automatically detect and alert authorities in real-time when crowd densities approach potentially critical levels in any given node

## Development

To install this repo, first install [`poetry`](https://python-poetry.org/docs/), then run `poetry install` on a `python>=3.10` environment.

The `motrackers` package has a slight bug. I've raised [an issue](https://github.com/adipandas/multi-object-tracker/issues/51) on their repo but not sure if it will get resolved. Until then, we should make this change manually on the locally installed package.

Please visit the [`SOMPT22`](https://sompt22.github.io/) (Surveillance Oriented Multi-Pedestrian Tracking Dataset) website to download the dataset used for the training of our Object Tracking and Object Detection Models. Additional information on the research that generated this dataset can be found [`here`](https://arxiv.org/abs/2208.02580). 

### Running

To run the docker containers, run `docker compose up --build` from the repo's base directory.

### Testing

To test, run `poetry run pytest` from the root directory.

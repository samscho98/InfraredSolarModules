# InfraredSolarModules ML model

A machine learning model based on an existing dataset. For more information see the dataset [InfraredSolarModules](https://github.com/RaptorMaps/InfraredSolarModules). 

## Installation

Unzip data/2020-02-14_InfraredSolarModules.zip and place the images folder in data/InfraredSolarModules or download the [dataset](https://github.com/RaptorMaps/InfraredSolarModules) and copy the images into /data/InfraredSolarModules/images.

## Usage

You can train the model by executing the main.py file with the command:

```bash
python main.py --mode train
```

You can evaluate the existing model with the command:
```bash
python main.py --mode evaluate
```

If you want to test a single image, run the following command:
```bash
python main.py --mode predict --image_path path/to/your/image.jpg
```

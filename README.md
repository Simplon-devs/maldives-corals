# maldives_corals package

This package contains the AI models needed to perform coral fragments and reef frame detection.

## Original work

This code is a deep refactoring from the code of Gaétan Morand available [here](https://github.com/morand-g/maldives-corals). Inhis repository, you would find the original explanations of his work, in the Readme available there. These are not adapted to the refactoring done here, explaning this new Readme that follows.


## Installation

In order to install the maldives_corals package, run the following commands:
```
git clone https://github.com/Simplon-devs/maldives-corals.git
cd maldives_corals

pip install .
```

Do not forget to create your own virtual environment if needed to use the package where you want to use it in your computer.

For the model to work, you will also need to download the pretrained model and its parameters. To that end, download [this folder](https://drive.google.com/drive/folders/1MluLeh9jHxo0CYyZgnXtsvgun5ISGrcT?usp=share_link) and paste its contents in the directory where you run your code. You also need an image you can paste in this same directory to launch the test script given below.

## Using the package

The package's models are available through the maldives_corals.models.CoralsModels class. The following models are currently available:
- A coral fagments detection model that detects the position of all coral fragments on an image, as well as their species and wether or not they are dead/bleached.
- A frame detection model that generates a mask showing a coral reef structure on an image.

Both of these models are currently under development.


```python
import numpy as np
from PIL import Image
from maldives_corals.models import CoralsModels

models = CoralsModels()
img = np.asarray(Image.open('image_test.jpg'))

# detect_corals takes a list of images as a parameter and returns the classes,
# probability percentages of each class and positions of the corals on the images
pred = models.detect_corals([img])

print(pred[0])
```

Output:
```
[['acropora', 96.04, 0.5618055555555556, 0.6268518518518519, 0.07013888888888889, 0.06666666666666667], ['acropora', 98.71, 0.4326388888888889, 0.6157407407407407, 0.08611111111111111, 0.046296296296296294], ...]
```

By default, the function detect_corals returns a list of annotations. If you want, you can call the function with the parameter ```return_images=True``` in order to get a list of image arrays instead:

```python
pred = models.detect_corals([img], return_images=True)

print(pred[0])
```

Output:
```
[[[  5  57  70]
  [  2  56  68]
  [  0  56  64]
  ...
  [  0 159 169]
  [  0 163 173]
  [  0 167 176]]]
```


You can also train the corals detection model yourself by providing it with a list of image RGB arrays and the corresponding annotations. The annotations must follow the YOLO format. See [the imageAI documentation](https://imageai.readthedocs.io/en/latest/customdetection/index.html) for more information.

```python
models = CoralsModels()
models.fit_corals_detection(images, annotations) 
```


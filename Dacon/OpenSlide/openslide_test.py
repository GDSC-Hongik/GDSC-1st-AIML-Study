from __future__ import division
OPENSLIDE_PATH = r'C:\openslide-win64-20221111\bin'

import os
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        from openslide import open_slide
        import openslide
        print('import 성공')
else:
    import openslide
    print('import 성공2')

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

slide = open_slide()


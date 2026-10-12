# Test assets

Two real photographs are used by the detection tests so that they run offline
and against genuine faces rather than synthetic shapes.

| File | Faces | Source |
| --- | --- | --- |
| `one_face.jpg` | 1 | OpenCV sample image `samples/data/lena.jpg` |
| `group.png` | ~58 | OpenCV cascade test image `testdata/cv/cascadeandhog/images/class57.png` |

`group.png` is a large class photograph whose faces span a wide range of
sizes, which makes it the right fixture for checking multi-face and small-face
detection.

If either file is missing, the tests that need it are skipped rather than
failed. Re-download them with:

```bash
curl -o tests/assets/one_face.jpg \
  https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg

curl -o tests/assets/group.png \
  https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/cv/cascadeandhog/images/class57.png
```

The remaining fixtures (a no-face gradient, a blank image, random noise, a
corrupted file and a three-face composite built from `one_face.jpg`) are
generated at test time in `tests/conftest.py`.

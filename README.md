# Mobile clicker game bot maded with [py android viewer](https://github.com/razumeiko/py-android-viewer) and [scrcpy](https://github.com/Genymobile/scrcpy)

## **!!!Program working only with Android phones!!!**

My part in this program is `__init__.py` file, one function in `control.py` (`tap`) and `images` folder with images needed to recognize things. It's old project and may not work now.

**Scrcpy** provides display and control of Android devices connected via USB, **py android viewer** allows to send touch events to android devices and uses scrcpy under the hood. I used this because game recognize if it is launched on Android emulator on PC.

1. Install dependencies from requirements.txt: `pip install -r requirements.txt`
2. You need turn on "developer mode" on phone
3. I don't remember, maybe install adb for your phone on your computer?
4. Connect your phone to your computer with a USB cable
5. Run program: `python __init__.py`

## py android viewer (from it's README.md file)

This package allows you to get video stream from android device.

Stream frames already converted to numpy array using [PyAV](https://github.com/mikeboers/PyAV)

Also this package allows you to send touch events to devices.

### How it works

I am using [scrcpy](https://github.com/Genymobile/scrcpy) server and connect to two sockets, video and control sockets.
Video stream has almost no delay because i am not using separate process of `ffmpeg` and use direct bindings to `ffmpeg` to decode video.

### Requirements

[PyAV](http://docs.mikeboers.com/pyav/develop/overview/installation.html)

[ffmpeg](http://ffmpeg.org/)

### Examples

```python
import cv2
from viewer import AndroidViewer

# This will deploy and run server on android device connected to USB
android = AndroidViewer()

while True:
    frames = android.get_next_frames()
    if frames is None:
        continue

    for frame in frames:
        cv2.imshow('game', frame)
        cv2.waitKey(1)
```

Send swipe event:

```python
    android = AndroidViewer()
    android.swipe(start_x, start_y, end_x, end_y)
```

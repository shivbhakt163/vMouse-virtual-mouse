# vMouse-virtual-mouse
## Description

vMouse is a Python application that uses OpenCV and MediaPipe for hand tracking to control the mouse. The project captures video from your webcam and detects hand movements to move the mouse cursor and perform click actions.

## Features

- Real-time hand tracking using MediaPipe
- Control mouse cursor with index finger movements
- Left-click and right-click actions based on finger distances

<h1>Installation</h1>

just download/clone the github repo from the link : 
<ol>
    <li>Clone the repository:</li>
</ol>

<pre><code>git clone https://github.com/shivbhakt163/vMouse-virtual-mouse.git
</code></pre>


### Prerequisites
<br>
-python-3.12.3-amd64.exe (python version 3.11 and up)
</br>
<br>
-"gateway_packages.txt" for the required modules and you'll be getting them in the directory.
</br>
<br>
-"pkg_installer.bat" for installation of the required modules, installation is simple just run the pkg_installer.bat.
</br>

## Usage

1. **Change the directory:**
   ```sh
   cd vMouse-virtual-mouse
   ```

2. **run the batch file:**
   ```sh
   pkg_installer.bat
   ``` 

3. **Run the Python script:**

    ```sh
    python main.py
    ```

4. **Control the mouse with your hand movements:**
    - Move the mouse cursor with your index finger.
    - Perform a left-click by bringing your index and middle fingers close together.
    - Perform a right-click by bringing your thumb and index finger close together.
    - Perform a drag and drop operation by bringing your thumb and index finger close to each other.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [OpenCV](https://opencv.org/)
- [MediaPipe](https://mediapipe.dev/)
- [pynput](https://pypi.org/project/pynput/)


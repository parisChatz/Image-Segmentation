[![Contributors][contributors-shield]][contributors-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h3 align="center">Image Segmentation</h3>

  <p align="center">
    This package is used for segmenting the cancerous lesions from close-up skin images. 
    <br />
    <a href="https://github.com/parisChatz/Image-Segmentation">View Demo</a>
    ·
    <a href="https://github.com/parisChatz/Image-Segmentation/issues">Report Bug</a>
    ·
    <a href="https://github.com/parisChatz/Image-Segmentation/issues">Request Feature</a>
  </p>




<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
  * [Installation](#installation)
  * [Executing](#executing)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)




<!-- ABOUT THE PROJECT -->
## About The Project
This is a project for the Computer Vision module of Robotics and Autonomous Systems MSc, University of Lincoln, UK.
The goal was for each skin lesion image to use image processing techniques to
automatically segment lesion object. Moreover, for each skin lesion image, the Dice Similarity Score was needed to be
calculated. 



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.


### Installation
 
Clone repository
```
git clone https://github.com/parisChatz/Image-Segmentation.git
```
Install requirements
```
sudo pip3 install -r requirements.txt
```

### Executing
After downloading a dataset and changing the paths.py, the package is ready. 
In the repository a small dataset is provided for testing and execution purposes.

There are 2 main executables:
* kmeans_segm.py
* otsu_segm.py

Example script execution:
```
python3 kmeans_segm.py
```

Both scripts go through the paths defined and check each image, segmenting the images accordingly.

[![Product Name Screen Shot][product-screenshot]](https://example.com)

The scripts continuously output the Dice Score for each image and the updated mean Dice Score.

[![Product Name Screen Shot][product-screenshot2]](https://example.com)


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- LICENSE -->
## License
Distributed under the MIT License. 



<!-- CONTACT -->
## Contact
Email: parischatz94@gmail.com


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/parisChatz/Image-Segmentation
[contributors-url]: https://github.com/parisChatz/Image-Segmentation/graphs/contributors

[issues-shield]: https://img.shields.io/github/issues-raw/parisChatz/Image-Segmentation
[issues-url]: https://github.com/parisChatz/Image-Segmentation/issues

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/paris-chatzithanos/

[product-screenshot]: images/segm_git.png
[product-screenshot2]: images/git2.png

const tf = require('@tensorflow/tfjs-node');
const Jimp = require('jimp');

// mapping of Fashion-MNIST labels
const labels = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
  ];

  const imageWidth = 28;
  const imageHeight =28;
  const imageChannels = 1;

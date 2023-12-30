
## Time invariant Self Organising Map (SOM) methods
Variation of the self-organising map algorithm where the original time-dependent (learning rate and neighbourhood) learning function is replaced by a time-invariant one. These time independent methods allow continous learning.

These implemetation allows to viusalize the convergence in input space. Hence it is advisable to use this for pattern in 1D, 2D, 3D space.


## References
- [Self Organising Maps - Teuvo Kohonen](https://link.springer.com/book/10.1007/978-3-642-97610-0)
- [DSOM - N. Rougier](https://www.sciencedirect.com/science/article/abs/pii/S0925231211000713?via%3Dihub)
- [A New Self-Organizing Map with Continuous Learning Capability - H. Hikawa](https://ieeexplore.ieee.org/document/8628891)
- [Brain-inspired self-organizing model for incremental learning - K. Gunawardena](https://ieeexplore.ieee.org/document/6706851/)
- [Improving Quantization Quality in Brain-Inspired Self-organization for Non-stationary Data Spaces - K. Gunawardena](https://link.springer.com/chapter/10.1007/978-3-319-12637-1_65)
- [PLSOM - E. Berglund](https://www.semanticscholar.org/paper/The-parameterless-self-organizing-map-algorithm-Berglund-Sitte/671ecfe9e8e0443eb2afcaeef823da8d69ba86a9)
[TASOM - H. Shah-Hosseini](https://ieeexplore.ieee.org/document/844265/)


## How to Use 

```
    data = np.loadtxt("data.csv", delimiter=",")
    som = SOM((15, 15), 2)
    som.learn(data, epochs=30, plot=True)
```
## Tech Stack

```
    numpy
    warnings
    matplotlib
    functools
    typing
    sys
```


## Demo

https://github.com/cots-lab/Continuous-Learing-SOM/assets/73787312/3d44fc66-93cd-4671-a9d2-1abbc4149bf4



## Acknowledgements

 - [DSOM - Dr. Rougier](https://github.com/rougier/dynamic-som)
 - [sklearn SOM](https://github.com/rileypsmith/sklearn-som)


## Contributing

Contributions are always welcome!

## License

[MIT](https://github.com/cots-lab/Continuous-Learing-SOM/blob/main/LICENSE)


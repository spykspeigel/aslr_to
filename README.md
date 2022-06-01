# ASR_TO (Trajectory Optimsation For Articulated Soft  Robots)
## A prototype code in python for optimal control problem of soft actuated robots.

* [crocodddyl](https://github.com/stack-of-tasks/pinocchio)
* [pinocchio](https://github.com/stack-of-tasks/pinocchio)
* [example-robot-data](https://github.com/gepetto/example-robot-data) (optional for examples, install Python loaders)
* [gepetto-viewer-corba](https://github.com/Gepetto/gepetto-viewer-corba) (optional for display)
* [matplotlib](https://matplotlib.org/) (optional for examples)

### Building from source
The code completely written in Python, however, we still need to compile it (using cmake) to be able to install, run examples and unittest.
The compilation is as simple as:
```bash
mkdir build & cd build
cmake ../
make
sudo make install
```

You can run examples, unit-tests and benchmarks from your build dir:

```bash
cd build
make test
```

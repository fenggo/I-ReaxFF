## Introductions for building a mathine learning potential

### Use the following command to run the train process
```bash
nohup ./train.py --f=1 --t=1 --s=10000 --z=1 > py.log 2>&1 &
```
* options:

--f: whether optimize the four-body parameter

--t: whether optimize the three-body parameter

--h: whether optimize the hydrogen-bond parameter

--z: whether evaluate the zero point energy

--s: the train step

### Add data to the current training data-set


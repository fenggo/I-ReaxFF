#!/usr/bin/env python
from irff.molecule import packmol




packmol(strucs=['h2o','h2o','h2o','h2o','h2o','h2o','h2o','h2o','h2o',
	            'ch4','h2o','ch4','ethane','ethane'],
        supercell=[3,3,2],
        w=True)


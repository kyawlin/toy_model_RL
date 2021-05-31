# Using policy gradient to control liquid level in connected tanks



# Image

![Semantic ddd of image](./tank.svg "Image")

## Problem Description

A series of tanks are connected with a valve between. There are inlet pipe and outlet pipe connected to the outermost tanks.
The goal is to keep the water level in the tanks as long as possible.



--- Environment --
The liquid level is controlled by the following equation.
<img src="https://render.githubusercontent.com/render/math?math=\frac{d(h)}{dt}=\frac{Q_{in}-VA_{pipe}\sqrt{2gh+\frac{\Delta P} {\rho}}}{\pi r^2}">

, where $h$ is the height of the tank
<img src="https://render.githubusercontent.com/render/math?math=r"> is radius of the tank,
<img src="https://render.githubusercontent.com/render/math?math=Q_{in}"> is inlet flow rate per second,
<img src="https://render.githubusercontent.com/render/math?math=V"> is valve action (either 1 or 0),
<img src="https://render.githubusercontent.com/render/math?math=A_{pipe}"> is pipe area,
<img src="https://render.githubusercontent.com/render/math?math=g">is gravity,
<img src="https://render.githubusercontent.com/render/math?math=\rho"> is liquid density,
<img src="https://render.githubusercontent.com/render/math?math=\Delta"> is pressure difference between tank surface and atmosphere.
Water flows into the first tank with a flow rate <img src="https://render.githubusercontent.com/render/math?math=\sim \mathcal{N}(q_{in,t},\sigma)"> 

Observation includes water level divided by tank height, water level difference to the previous level, whether tank is half-fill and whether valve is opened.
Reward is -5 if water level is outside the limits otherwise it is one.
If one tank's level is outside the limits, then the episode is done.

## requirements
Pytorch, Numpy









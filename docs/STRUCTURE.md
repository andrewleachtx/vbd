# Structure

### Goals
1. Render scenes exhibiting extreme position constraints, and visualize them being solved taking advantage of the "maximized parallelism" design in the paper, which leverages GPU to parallelize vertex updates for a given color. To start, I want two scenes - one with a dragon being subject to insane constraints, and then releasing them, and then one where something falls into a teapot.
   1. I will start with no rendering, writing to file data to be visualized like [Blender](https://www.blender.org/).
   2. After observing the times with certain scenes I may be able to leverage Polyscope or something else to continously render the scene.
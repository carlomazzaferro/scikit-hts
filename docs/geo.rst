Geo Handling Capabilities
=========================



In our use case for germany, the true format is the following:


    Level                                   Node Key                           # of nodes

      1                                     germany                                 1

      2                 berlin              hamburg            Munich      ...      5

      3          hex1  hex2   ...      hex7  hex8  ...     hex11  hex12 ...   ...  114

      4                            .....     hex 7 level    .....                  520


As you can see the number of nodes increases significantly with each level. However, many of these nodes will have barely any observations, as they simply exist due to the fact at _some point_ in the timespan considered a ride happen to start there.

In order to make ther problem tractable, and to be able to actually perform predictions/forecasting at the levels desired, the tree building  data structure accepts a `min_count` parameter, which will remove any node having less than the desired number of observations.
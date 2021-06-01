# -------------------------------------------------------------------------------
# Description:  pprint的例子
# Reference:
# Name:   pprint_example
# Author: wujun
# Date:   2021/4/28
# -------------------------------------------------------------------------------

import pprint
data = (
    "this is a string",
    [1, 2, 3, 4],
    ("more tuples", 1.0, 2.3, 4.5),
    "this is yet another string"
    )

print(data)
pprint.pprint(data)

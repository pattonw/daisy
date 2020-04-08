import daisy
import copy


Coordinate = daisy.Coordinate
x = Coordinate([0, 1, 2])
print(Coordinate.__len__(x))
print(Coordinate.__getitem__(x, 0))
print(len(x))
print(x[0])
print(x.dims())

z = copy.deepcopy(x)
print(list(x))
y = Coordinate([2, 1, 0])
w = copy.deepcopy(y)
assert z + w == Coordinate([2, 2, 2]), f"{z+w}"
assert z - w == Coordinate([-2, 0, 2]), f"{z-w}"
assert z < w


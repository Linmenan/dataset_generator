import math


class Vector2D:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    @classmethod
    def from_points(cls, start_point, end_point):
        return cls(end_point.x - start_point.x, end_point.y - start_point.y)

    def angle(self):
        return math.atan2(self.y, self.x)

    def normal_vector(self) -> "Vector2D":
        return Vector2D(-self.y, self.x)

    def norm(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def normalize_self(self):
        norm = self.norm()
        self.x /= norm
        self.y /= norm

    def inner_product(self, vector: "Vector2D"):
        return self.x * vector.x + self.y * vector.y

    def projection(self, vector: "Vector2D"):
        return self.inner_product(vector) / vector.norm()

    def cross_product_to(self, vector: "Vector2D"):
        return self.x * vector.y - self.y * vector.x



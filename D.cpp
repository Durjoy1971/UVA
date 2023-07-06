1. Distance Formula:
   - Euclidean Distance: d = sqrt((x2 - x1)^2 + (y2 - y1)^2)
   - Manhattan Distance: d = |x2 - x1| + |y2 - y1|

2. Slope of a Line:
   - Slope (m) = (y2 - y1) / (x2 - x1)

3. Area of a Triangle:
   - Given three points A(x1, y1), B(x2, y2), C(x3, y3):
   - Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|

4. Pythagorean Theorem:
   - In a right triangle with sides a, b, and hypotenuse c:
   - c^2 = a^2 + b^2

5. Circle Formulas:
   - Circumference: C = 2 * π * r
   - Area: A = π * r^2

6. Polygon Area:
   - Shoelace Formula:
     - Given n points (x[i], y[i]) in clockwise or counterclockwise order:
     - Area = 0.5 * |(x[0]*y[1] + x[1]*y[2] + ... + x[n-1]*y[0]) - (x[1]*y[0] + x[2]*y[1] + ... + x[0]*y[n-1])|

7. Intersection of Line Segments:
   - To check if two line segments AB and CD intersect:
   - Check if (A, B, C) and (A, B, D) have opposite orientations, and (C, D, A) and (C, D, B) have opposite orientations.
struct Point {
    int x, y;
};

// Function to find the orientation of three points (p, q, r)
int orientation(Point p, Point q, Point r) {
    int val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);

    if (val == 0)
        return 0; // collinear
    else if (val > 0)
        return 1; // clockwise
    else
        return 2; // counterclockwise
}

// Function to check if two line segments intersect
bool doSegmentsIntersect(Point p1, Point q1, Point p2, Point q2) {
    int o1 = orientation(p1, q1, p2);
    int o2 = orientation(p1, q1, q2);
    int o3 = orientation(p2, q2, p1);
    int o4 = orientation(p2, q2, q1);

    if (o1 != o2 && o3 != o4)
        return true; // Intersection occurs

    // Special cases for collinear and overlapping segments
    if (o1 == 0 && onSegment(p1, p2, q1))
        return true;

    if (o2 == 0 && onSegment(p1, q2, q1))
        return true;

    if (o3 == 0 && onSegment(p2, p1, q2))
        return true;

    if (o4 == 0 && onSegment(p2, q1, q2))
        return true;

    return false; // No intersection
}

8. Convex Hull:
   - The Convex Hull of a set of points is the smallest convex polygon that contains all the points.
   - Graham's Scan or Jarvis's Algorithm can be used to compute the Convex Hull efficiently.

9. Line Intersection:
   - Given two lines with equations y = m1 * x + c1 and y = m2 * x + c2:
   - The lines intersect at (x, y), where x = (c2 - c1) / (m1 - m2) and y = m1 * x + c1.

10. Circle-Line Intersection:
    - Given a circle with center (a, b) and radius r, and a line with equation y = mx + c:
    - Substitute the equation of the line into the equation of the circle and solve for x to find the intersection points.

various variations of quadrilaterals:

1. Square:
   - A square is a quadrilateral with all sides equal in length and all angles equal to 90 degrees.
   - Perimeter: P = 4s, where s is the length of a side.
   - Area: A = s^2, where s is the length of a side.
   - Diagonal: d = s * sqrt(2), where s is the length of a side.

2. Rectangle:
   - A rectangle is a quadrilateral with all angles equal to 90 degrees.
   - Perimeter: P = 2(l + w), where l and w are the lengths of the adjacent sides.
   - Area: A = l * w, where l and w are the lengths of the adjacent sides.
   - Diagonal: d = sqrt(l^2 + w^2), where l and w are the lengths of the adjacent sides.

3. Rhombus:
   - A rhombus is a quadrilateral with all sides equal in length.
   - Perimeter: P = 4s, where s is the length of a side.
   - Area: A = (d1 * d2) / 2, where d1 and d2 are the lengths of the diagonals.
   - Diagonal: d = sqrt((s1^2 + s2^2) / 2), where s1 and s2 are the lengths of the adjacent sides.

4. Parallelogram:
   - A parallelogram is a quadrilateral with opposite sides parallel.
   - Perimeter: P = 2(a + b), where a and b are the lengths of the adjacent sides.
   - Area: A = b * h, where b is the length of the base and h is the height.
   - Diagonals bisect each other.

5. Trapezoid (Trapezium):
   - A trapezoid is a quadrilateral with one pair of parallel sides.
   - Perimeter: P = a + b + c + d, where a, b, c, and d are the lengths of the sides.
   - Area: A = (h/2) * (a + b), where h is the height and a, b are the lengths of the parallel sides.
   - Diagonals do not necessarily have any special properties.

Remember to use the appropriate formulas based on the properties and given information of the specific quadrilateral you are working with.

#include <cmath>

const double pi = 3.14159265358979323846;

// Convert degrees to radians
double toRadians(double degrees) {
    return degrees * pi / 180.0;
}

// Calculate sine of an angle in degrees
double sinDegrees(double degrees) {
    double radians = toRadians(degrees);
    return sin(radians);
}

// Calculate cosine of an angle in degrees
double cosDegrees(double degrees) {
    double radians = toRadians(degrees);
    return cos(radians);
}

// Calculate tangent of an angle in degrees
double tanDegrees(double degrees) {
    double radians = toRadians(degrees);
    return tan(radians);
}

// Calculate cotangent of an angle in degrees
double cotDegrees(double degrees) {
    double radians = toRadians(degrees);
    return 1.0 / tan(radians);
}

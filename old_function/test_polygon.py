import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, Point

def draw_polygon_with_point(polygon_points, point, polygon_color='blue', point_color='red'):
    """
    Draws a polygon using 6 points and marks another point with a different color.

    :param polygon_points: List of 6 (x, y) tuples representing the polygon vertices.
    :param point: A tuple (x, y) representing the additional point.
    :param polygon_color: Color of the polygon.
    :param point_color: Color of the additional point.
    """
    if len(polygon_points) != 6:
        raise ValueError("You must provide exactly 6 points for the polygon.")
    
    # Create a polygon and add the first point again to close the shape
    polygon = Polygon(polygon_points)
    
    # Extract x and y coordinates for plotting
    x, y = zip(*polygon.exterior.coords)
    
    # Plot polygon
    plt.plot(x, y, color=polygon_color, linewidth=2, label='Polygon')
    
    # Plot point
    plt.scatter(*point, color=point_color, marker='o', s=100, label='Point')
    
    # Labels & display
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Polygon with a Highlighted Point')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # Keep aspect ratio
    plt.show()

# Example usage
polygon_pts = [( 952.43,      572.03),(   1024.3,      575.54), ( 1090.4,      651.68), (1132.7,      756.91),( 837.02,      752.86), ( 896.03,      656.51)]
point = (995.5 ,707)
draw_polygon_with_point(polygon_pts, point)

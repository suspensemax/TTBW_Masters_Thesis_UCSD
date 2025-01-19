import numpy as np

vertex05_left_strut = np.array([60.115, -49.943, 4.265]) # left leading 6
point40_wing = np.array([56.172, -42.646, 5.821]) # * ft2m # Left leading edge 2

def calculate_angle(point1, point2):
    # Calculate the vectors from the origin to each point
    vector1 = np.array(point1)
    vector2 = np.array(point2)
    
    # Calculate the dot product and magnitudes of the vectors
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    # Calculate the angle in radians
    angle_radians = np.arccos(dot_product / (magnitude1 * magnitude2))
    
    # Convert the angle to degrees
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees

# Example points
point1 = point40_wing
point2 = vertex05_left_strut

# Calculate the angle between the two points
angle = calculate_angle(point1, point2)
print(f"The angle between the points is {angle:.2f} degrees.")
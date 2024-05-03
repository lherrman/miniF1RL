import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from enum import Enum
from pprint import pprint
import numpy as np
import math
import pygame as pg
from pygame.math import Vector2

from config import Config as cfg
import gymnasium
from gymnasium import spaces

import cv2
import matplotlib.pyplot as plt
import numpy as np


CAR_START_POSITION = (3, 10) # Needs to be adjusted to the track
TRACK_IMAGE_PATH = "track.png"

# Parts of this class are based on the CarModel class i implemented for another project (SlamCar)
# (base_model.py) https://github.com/lherrman/slamcar-controller

class CarModel:
    def __init__(self, x, y):
        # Car parameters are loaded from the config file
        self.length = 0             # length of the car in meters
        self.width = 0              # width of the car in meters
        self.max_velocity = 0       # meters per second
        self.acceleration_speed = 0 # meters per second
        self.steering_speed =   0   # degrees per second
        self.max_steering = 0       # degrees
        self._load_parameters_from_config()

        # Car state
        self.position = Vector2(x, y)                # position in meters
        self.heading = Vector2(0, -1).normalize()    # heading vector
        self.velocity = Vector2(0.0, 0.0)            # velocity
        self.velocity_magnitude = 0.0                # velocity magnitude
        self.steering = 0.0                          # tire angle in degrees
        self.collision = False                       # collision flag

        # Camera state
        self.camera_position = Vector2(0, 0) # position of the camera in the world
        self.camera_position_smooth = Vector2(0, 0) # smoothed position of the camera in the world

        self.ppu = 128 # pixels per unit

        # Load track boundaries from image and initialize lidar sensor vectors
        self.track_boundaries = self._get_track_boundaries_from_image(TRACK_IMAGE_PATH)
        self._init_lidar_sensor_vectors([-45, 0, 45])

    def _init_lidar_sensor_vectors(self, sensor_angles):
        self.lidar_sensor_vectors = []
        for i in sensor_angles:
            angle = math.radians(i)
            self.lidar_sensor_vectors.append(Vector2(math.cos(angle), math.sin(angle)))
        self.lidar_sensor_distances = [0] * len(self.lidar_sensor_vectors)
        self.current_sensor_vectors = [] # stores the current sensor vectors rotated according to the car heading

    def _get_track_boundaries_from_image(self, image_path) -> dict:
        '''
        Using opencv to read the image and find the contours of the track boundaries.
        The boundaries are represented as list of points. ([x1, y1], [x2, y2], ...)
        '''
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        contours, _ = cv2.findContours(image.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        assert len(contours) == 2, "There should be exactly 2 contours in the image"

        # Transform and downsample the contours
        downsample_rate = 10
        inner = contours[0][:, 0, :][::downsample_rate]
        outer = contours[1][:, 0, :][::downsample_rate]

        # # Append last point to close the loop
        inner = np.append(inner, [inner[0]], axis=0)
        outer = np.append(outer, [outer[0]], axis=0)

        # Scale the contours to make suitable for the simulation
        inner, outer = inner / 60, outer / 60
        return {
            'inner': inner,
            'outer': outer
        }   

    def get_observation(self):
        # Returns the observation of the car (lidar sensor data)
        return self.lidar_sensor_distances

    def get_termination(self):
        # Returns the termination condition of the car (collision)
        return self.collision

    def _load_parameters_from_config(self):
        self.length = cfg.get("car_length")
        self.width = cfg.get("car_width")
        self.max_steering = cfg.get("max_steering")
        self.max_velocity = cfg.get("max_velocity")
        self.acceleration_speed = cfg.get("acceleration")
        self.steering_speed = cfg.get("steering_speed")
        self.drag_coefficient = cfg.get("drag_coefficient") # Currently not used
        self.slipage_coefficient = cfg.get("slipage_coefficient") # Currently not used
        self.drifting_coefficient = cfg.get("drifting_coefficient") # Currently not used

    def update(self, dt):
        #self._update_inputs_human(dt)
        self._update_position(dt)
        self._update_lidar_data()
        self._update_track_collision()
  
    def _update_position(self, dt):
        # Apply steering
        steering_angle = math.radians(self.steering * self.velocity_magnitude * 20)  # Adjust as needed
        self.heading = self.heading.rotate(steering_angle)

        # Apply drag
        drag_force = -self.drag_coefficient * self.velocity_magnitude * self.velocity_magnitude
        self.velocity_magnitude += drag_force * dt

        # Calculate velocity vector
        self.velocity = self.heading * self.velocity_magnitude

        # Apply drifting effect
        heading_perpendicular = self.heading.rotate(math.copysign(90, self.steering)).normalize()
        self.velocity += heading_perpendicular * self.drifting_coefficient * abs(self.steering * self.velocity_magnitude)

        # Update position
        self.position += self.velocity * dt

    def _update_inputs_human(self, dt):
        pressed = pg.key.get_pressed()

        # Steering
        self.update_steering(pressed[pg.K_a], pressed[pg.K_d], dt)

        # Velocity
        self.update_velocity(pressed[pg.K_w], pressed[pg.K_s], dt)

    def update_steering(self, left, right, dt):
        if left:
            self.steering = max(-self.max_steering, self.steering - self.steering_speed * dt)

        if right:
            self.steering = min(self.max_steering, self.steering + self.steering_speed * dt)

        # If neither is pressed, reduce steering to 0
        # back_steer_factor determines how fast the steering will go back to 0
        back_steer_factor = 2
        if not left and not right:
            if self.steering > 0:
                self.steering = max(0, self.steering - self.steering_speed * dt * back_steer_factor)
            else:
                self.steering = min(0, self.steering + self.steering_speed * dt * back_steer_factor)
        
    def update_velocity(self, up: bool, down: bool, boost: bool, dt: float):
        # Boost increases the max velocity and acceleration speed
        max_velocity = self.max_velocity if not boost else self.max_velocity * 3
        acceleration_speed = self.acceleration_speed if not boost else self.acceleration_speed * 3

        # Update velocity
        if up:
            self.velocity_magnitude = min(max_velocity, 
                                          self.velocity_magnitude + acceleration_speed * dt)
        if down:
            self.velocity_magnitude = max(0, 
                                          self.velocity_magnitude - acceleration_speed * dt)
            
        # If neither is pressed, reduce velocity to 0
        if not up and not down and not boost:
            if self.velocity_magnitude > 0:
                self.velocity_magnitude = max(0, self.velocity_magnitude - self.acceleration_speed * dt)
            else:
                self.velocity_magnitude = min(0, self.velocity_magnitude + self.acceleration_speed * dt)


    def _update_track_collision(self):
        '''Check if the car is colliding with the track boundaries'''
        # Simple approach to just check if any sensor is below a certain threshold
        sensors_min = min(self.lidar_sensor_distances)
        self.collision = sensors_min < 0.2

    def _update_camera_position(self, screen):
        self.camera_position = self.position - (Vector2(*screen.get_rect().center) / self.ppu)
        self.camera_position_smooth = self.camera_position_smooth * 0.97 + self.camera_position * 0.03

    def _update_lidar_data(self):
        '''Update the lidar sensor data by raycasting into the track boundaries'''
        self.current_sensor_vectors = []
        for i, sensor_vector in enumerate(self.lidar_sensor_vectors):
            sensor_vector = sensor_vector.rotate(-self.heading.angle_to(Vector2(1, 0)))
            self.current_sensor_vectors.append(sensor_vector)
            self.lidar_sensor_distances[i] = self._raycast_distance(self.position, sensor_vector, 50)

    def _raycast_distance(self, position, direction, max_distance):
        '''
        Shoot a ray from position in direction for max_distance into the track boundaries
        Return the distance to the first intersection point
        '''
        intersection_point = self._raycast(position, direction, max_distance)
        if intersection_point is None:
            return max_distance
        return (intersection_point - position).length()

    def _raycast(self, position, direction, max_distance):
        '''
        Shoot a ray from position in direction for max_distance into the track boundaries
        Return the intersection point
        '''
        # Implemented by ChatGPT
        # Convert position and direction to Vector2
        position = Vector2(position[0], position[1])
        direction = Vector2(direction[0], direction[1]).normalize()

        # Initialize variables to keep track of the nearest intersection point and its distance
        nearest_intersection = None
        nearest_distance = float('inf')

        # Iterate over each boundary segment
        for boundary in self.track_boundaries.values():
            boundary_points = [Vector2(p[0], p[1]) for p in boundary]
            for i in range(len(boundary_points) - 1):
                p1 = boundary_points[i]
                p2 = boundary_points[i + 1]

                # Check if the ray intersects with this line segment
                intersection_point = self._segment_intersection(position, position + direction * max_distance, p1, p2)
                if intersection_point is not None:
                    # Calculate the distance between the position and the intersection point
                    distance = (intersection_point - position).length()

                    # Update the nearest intersection point if this intersection is closer
                    if distance < nearest_distance:
                        nearest_intersection = intersection_point
                        nearest_distance = distance

        # Return the nearest intersection point
        return nearest_intersection

    def _get_neares_point_from_inner_boundary(self, position):
        # Get the nearest point from the inner boundary
        # Implemented by ChatGPT
        nearest_point = None
        nearest_distance = float('inf')
        for point in self.track_boundaries['inner']:
            distance = (position - Vector2(*point)).length()
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_point = Vector2(*point)
        return nearest_point

    def _segment_intersection(self, p1, p2, p3, p4):
        # Function to find the intersection point of two line segments
        # Implemented by ChatGPT
        def _ccw(A, B, C):
            return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)

        def _intersect(A, B, C, D):
            return _ccw(A,C,D) != _ccw(B,C,D) and _ccw(A,B,C) != _ccw(A,B,D)

        if _intersect(p1, p2, p3, p4):
            # Calculate the intersection point
            a1 = p2.y - p1.y
            b1 = p1.x - p2.x
            c1 = a1 * p1.x + b1 * p1.y
            
            a2 = p4.y - p3.y
            b2 = p3.x - p4.x
            c2 = a2 * p3.x + b2 * p3.y
            
            det = a1 * b2 - a2 * b1
            
            if det == 0:
                return None  # Parallel lines
            else:
                x = (b2 * c1 - b1 * c2) / det
                y = (a1 * c2 - a2 * c1) / det
                return Vector2(x, y)
        else:
            return None  # No intersection
        
    def draw(self, screen, ppu):
        self.ppu = ppu

        self._update_camera_position(screen)
        self._draw_grid(screen)
        self._draw_track_boundaries(screen)
        self._draw_tires(screen)
        self._draw_lidar(screen)
        self._draw_car(screen)

    def _draw_lidar(self, screen):
        for i, sensor_vector in enumerate(self.current_sensor_vectors):
            start_pos = (self.position - self.camera_position_smooth) * self.ppu
            end_pos = (self.position + sensor_vector * self.lidar_sensor_distances[i] - self.camera_position_smooth) * self.ppu
            pg.draw.line(screen, (0, 255, 0), start_pos, end_pos, 1)
            
    def _draw_track_boundaries(self, screen):
        for boundary in self.track_boundaries.values():
            for i in range(len(boundary) - 1):
                p1 = Vector2(*boundary[i])
                p2 = Vector2(*boundary[i + 1])
                p1 = (p1 - self.camera_position_smooth) * self.ppu
                p2 = (p2 - self.camera_position_smooth) * self.ppu
                if (p1.x < -300 or p2.x < -300 or p1.x > screen.get_width() + 300 or
                    p2.x > screen.get_width()+300 or
                    p1.y < -300 or p2.y < -300 or p1.y > screen.get_height() + 300 or
                    p2.y > screen.get_height()+300):
                    continue
                pg.draw.line(screen, (255, 0, 0), p1, p2, 2)

    def _draw_grid(self, screen):
        canvas_width = screen.get_width()
        canvas_height = screen.get_height()
    
        # TODO: with this method about 1/2 of the grid is drawn off screen
        start_x = int(self.camera_position_smooth.x * self.ppu) - canvas_width
        start_y = int(self.camera_position_smooth.y * self.ppu) - canvas_height
        stop_x = int(self.camera_position_smooth.x * self.ppu) + canvas_width
        stop_y = int(self.camera_position_smooth.y * self.ppu) + canvas_height
        start_x = start_x - start_x % self.ppu
        start_y = start_y - start_y % self.ppu

        for x in range(start_x, stop_x, self.ppu):
            pg.draw.line(screen, (50, 50, 50), (x - self.camera_position_smooth.x * self.ppu, 0), (x - self.camera_position_smooth.x * self.ppu, canvas_height))
        for y in range(start_y, stop_y, self.ppu):
            pg.draw.line(screen, (50, 50, 50), (0, y - self.camera_position_smooth.y * self.ppu), (canvas_width, y - self.camera_position_smooth.y * self.ppu))

    def _draw_car(self, screen):
        # draw rectangle representing car
        angle = -self.heading.angle_to(Vector2(1, 0))

        factor_overlap = 1.5 # determines how much the car goes over the tires from the front and back
        car_corner_points = [
            Vector2(-self.length / factor_overlap, self.width / 2),
            Vector2(self.length / factor_overlap, self.width / 2),
            Vector2(self.length / factor_overlap, -self.width / 2),
            Vector2(-self.length / factor_overlap, -self.width / 2)
        ]

        car_corner_points = [p.rotate(angle) for p in car_corner_points]
        car_corner_points = [p + self.position for p in car_corner_points]
        car_corner_points = [p - self.camera_position_smooth for p in car_corner_points]
        car_corner_points = [p * self.ppu for p in car_corner_points]

        pg.draw.polygon(screen, (255, 255, 255), car_corner_points, 0)

        # indicate front of car
        #pg.draw.line(screen, (255, 255, 0), self.position *  self.ppu, (self.position + self.heading) * self.ppu * 0.1, 1)

    def _draw_tires(self, screen):
        # Draw tires
        angle = -self.heading.angle_to(Vector2(1, 0))
        tire_width = self.width / 4
        tire_length = self.length / 4

        # Draw front tires
        tire_corner_points = [
            Vector2(-tire_length / 2, tire_width / 2),
            Vector2(tire_length / 2, tire_width / 2),
            Vector2(tire_length / 2, -tire_width / 2),
            Vector2(-tire_length / 2, -tire_width / 2)
        ]

        tire_corner_points = [p.rotate(angle + self.steering) for p in tire_corner_points]
        tire_corner_points = [p * self.ppu for p in tire_corner_points]

        tires = [
            Vector2(self.length / 2, self.width / 2),
            Vector2(self.length / 2, -self.width / 2),
        ]

        tires = [p.rotate(angle) for p in tires]
        tires = [p + self.position for p in tires]
        tires = [p - self.camera_position_smooth for p in tires]
        tires = [p * self.ppu for p in tires]

        for tire in tires:
            points = [p + tire for p in tire_corner_points]
            pg.draw.polygon(screen, (255, 255, 255), points, 2)

        # Draw rear tires
        tire_corner_points = [
            Vector2(-tire_length / 2, tire_width / 2),
            Vector2(tire_length / 2, tire_width / 2),
            Vector2(tire_length / 2, -tire_width / 2),
            Vector2(-tire_length / 2, -tire_width / 2)
        ]
        tire_corner_points = [p.rotate(angle) for p in tire_corner_points]
        tire_corner_points = [p * self.ppu for p in tire_corner_points]

        tires = [
            Vector2(-self.length / 2, self.width / 2),
            Vector2(-self.length / 2, -self.width / 2)
        ]
        tires = [p.rotate(angle) for p in tires]
        tires = [p + self.position for p in tires]
        tires = [p - self.camera_position_smooth for p in tires]
        tires = [p * self.ppu for p in tires]

        for tire in tires:
            points = [p + tire for p in tire_corner_points]
            pg.draw.polygon(screen, (255, 255, 255), points, 2)

class RenderMode(str, Enum):
    HUMAN = 'human'
    
class MiniF1RLEnv(gymnasium.Env):

    def __init__(self, render_mode: str = RenderMode.HUMAN):
        # Initialize pygame stuff
        self.render_mode = render_mode
        self.screen: pg.Surface|None = None
        self.clock = None
        # Initialize car model
        self.car = CarModel(*CAR_START_POSITION)
        # Initialize environment
        self.action_space = spaces.Discrete(3) # 0 = nothing, 1 = left, 2 = right, 3 = boost
        self.observation_space = spaces.Box(low=0, high=100, shape=(3,), dtype=np.float32)

    def step(self, action):
        dt = 1/60

        # For Simplicity, car will always move forward

        # Update car model
        match action:
            case 0:  # Nothing
                self.car.update_velocity(True, False, False, dt)
                self.car.update_steering(False, False, dt)
            case 1:  # Left
                self.car.update_velocity(True, False, False, dt)
                self.car.update_steering(True, False, dt)
            case 2:  # Right
                self.car.update_steering(False, True, dt)
                self.car.update_velocity(True, False, False, dt)
            case 3:  # Boost
                self.car.update_velocity(True, False, True, dt)
                self.car.update_steering(False, False, dt)
            case _:
                raise ValueError(f"Invalid action {action}")
            
        terminate = self.car.get_termination()

        if terminate:
            self.reset()

        observation = self.car.get_observation()
        
        self.car.update(dt)
        if self.render_mode == RenderMode.HUMAN:
            self.render()
        
    def reset(self):
        self.car = CarModel(*CAR_START_POSITION)
        return self.car.position

    def render(self, mode=RenderMode.HUMAN):
        if self.screen is None and mode == RenderMode.HUMAN:
            pg.init()
            pg.display.init()
            self.screen = pg.display.set_mode((800, 600))
            pg.display.set_caption("MiniF1RL")

        if self.clock is None:
            self.clock = pg.time.Clock()

        self.screen.fill((0, 0, 0))
        self.car.draw(self.screen, 64)
        pg.display.flip()

    def close(self):
        pg.quit()

    def seed(self, seed=None):
        pass

    def _draw_grid(self, screen):
        canvas_width = screen.get_width()
        canvas_height = screen.get_height()

if __name__ == '__main__':
    env = MiniF1RLEnv()
    env.reset()
    env.render()
    done = False

    keycontrolls = {"left": False, "right": False, "boost": False}
    while not done:

        # Update controlls
        for event in pg.event.get():
            if event.type == pg.QUIT:
                done = True

            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_a:
                    keycontrolls["left"] = True
                elif event.key == pg.K_d:
                    keycontrolls["right"] = True
                elif event.key == pg.K_w:
                    keycontrolls["boost"] = True
            elif event.type == pg.KEYUP:
                if event.key == pg.K_a:
                    keycontrolls["left"] = False
                elif event.key == pg.K_d:
                    keycontrolls["right"] = False
                elif event.key == pg.K_w:
                    keycontrolls["boost"] = False

        action = 0
        if keycontrolls["left"]:
            action = 1
        if keycontrolls["right"]:
            action = 2
        if keycontrolls["boost"]:
            action = 3

        print(action)
        env.step(action)
        env.render()
        env.clock.tick(60)
    env.close()
from datetime import datetime
import glob
import os
from pathlib import Path
import time
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


CAR_START_POSITION = (3.5, 10) # Needs to be adjusted to the track
DEFAULT_TRACK_INDEX = 1

# Parts of this class are based on the CarModel class i implemented for another project (SlamCar)
# (base_model.py) https://github.com/lherrman/slamcar-controller

class CarModel:
    def __init__(self, x, y):
        # Car parameters are loaded from the config file
        self._load_parameters_from_config()

        # Car / Game state
        self.position = Vector2(x, y)                # position in meters
        self.heading = Vector2(0, -1).normalize()    # heading vector
        self.velocity = Vector2(0.0, 0.0)            # velocity
        self.velocity_magnitude = 0.0                # velocity magnitude
        self.steering = 0.0                          # tire angle in degrees
        self.collision = False                       # collision flag
        self.progress = 0.0                          # progress of the car on the track
        self.last_progress = 0.0                     # last progress of the car on the track

        # Camera state
        self.camera_position = Vector2(0, 0) # position of the camera in the world
        self.camera_position_smooth = Vector2(0, 0) # smoothed position of the camera in the world

        self.ppu = 64 # pixels per unit

        # Load track boundaries from image and initialize lidar sensor vectors
        self.available_tracks = self._get_available_tracks()
        self.current_track_index = DEFAULT_TRACK_INDEX
        self.switch_track(self.current_track_index)

        self.progress_boundary = self._calculate_track_progress_boundary() # Used to calculate the progress of the car
        self._init_lidar_sensor_vectors([-45, 0, 45])
        self.lidar_max_distance = 50
        self.valid_position_bbox = ((-50, 50), (-50, 50))

        # Reward Weights
        self.W1_progress = 1000
        self.W2_speed = 1.0
        self.W3_finish = 100
        self.W4_collision = -200

        self.graphics_modes = ['wireframe', 'graphics']
        self.graphics_mode = 'wireframe'
        self.controlls_buffer = {"left": False, "right": False, "up": False, "down": False, "boost": False}

        self.reset()

    def _load_parameters_from_config(self):
        self.length = cfg.get("car_length")
        self.width = cfg.get("car_width")
        self.max_steering = cfg.get("max_steering")
        self.max_velocity = cfg.get("max_velocity")
        self.acceleration_speed = cfg.get("acceleration")
        self.steering_speed = cfg.get("steering_speed")
        self.boost_factor = cfg.get("boost_factor")
        self.drag_coefficient = cfg.get("drag_coefficient") # Currently not used
        self.slipage_coefficient = cfg.get("slipage_coefficient") # Currently not used
        self.drifting_coefficient = cfg.get("drifting_coefficient") # Currently not used

    def get_observation(self):
        # Returns the observation of the car (lidar sensor data) and the steering angle
        return np.array(self.lidar_sensor_distances + [self.steering])

    def get_termination(self):
        car_invalid_position = (not self.valid_position_bbox[0][0] < self.position.x < self.valid_position_bbox[0][1] 
                                or not self.valid_position_bbox[1][0] < self.position.y < self.valid_position_bbox[1][1])
        return self.collision or car_invalid_position

    def get_truncate(self):
        # Returns the termination condition of the car (collision or reaching the end of the track)
        return self.progress > 0.99
    
    def get_reward(self):
        
        progress_diff_reward = self.progress - self.last_progress

        speed_reward = self.velocity_magnitude / self.max_velocity

        finish_reward = 1 if self.progress > 0.99 else 0

        collision_reward = 1 if self.collision else 0

        weighted_progress_diff_reward = self.W1_progress * progress_diff_reward
        weighted_speed_reward = self.W2_speed * speed_reward
        weighted_finish_reward = self.W3_finish * finish_reward
        weighted_collision_reward = self.W4_collision * collision_reward
        reward = (
                    weighted_progress_diff_reward +
                    weighted_speed_reward +
                    weighted_finish_reward +
                    weighted_collision_reward
                  )

        self.last_progress = self.progress
        return reward

    def reset(self, randomize=True):
        
        def get_random_Start_position():
            random_inner_boundary_index = np.random.randint(1, len(self.track_boundaries['inner']) - 1)
            p1 = Vector2(*self.track_boundaries['inner'][random_inner_boundary_index - 1])
            p2 = Vector2(*self.track_boundaries['inner'][random_inner_boundary_index])
            heading = p2 - p1
            nearest_point_on_outer_boundary = self._get_neares_point_from_boundary(p1, boundary='outer')
            p1_to_nearest_outer = nearest_point_on_outer_boundary - p1
            start_position = p1 + (p1_to_nearest_outer / 2)
            return start_position, -heading.normalize()

        self.position, self.heading = get_random_Start_position()
        self.velocity = Vector2(0.0, 0.0)
        self.velocity_magnitude = 0.0
        self.steering = 0.0
        self.collision = False
        self.progress = 0.0
        self.last_progress = 0.0
        self._load_parameters_from_config()
        self.progress_boundary = self._calculate_track_progress_boundary()

    def switch_graphics_mode(self):
        self.graphics_mode = self.graphics_modes[(self.graphics_modes.index(self.graphics_mode) + 1) % len(self.graphics_modes)]

    def zoom(self, factor):
        self.ppu = max(10, self.ppu + factor)
        scale_factor = self.ppu / 60
        self.background_image_scaled = pg.transform.scale(self.background_image, (int(self.background_image.get_width() * scale_factor), int(self.background_image.get_height() * scale_factor)))
   
    def switch_track(self, track_index=None):
        if track_index is None:
            index = (self.current_track_index + 1) % len(self.available_tracks)
            self.current_track_index = index
            self._set_track(self.available_tracks[index])
        else:
            track_index = min(track_index, len(self.available_tracks) - 1)
            self.current_track_index = track_index
            self._set_track(self.available_tracks[track_index])
        self.reset()

    def _get_available_tracks(self):
        all_track_files = glob.glob("assets/track*.png")
        available_tracks = [file for file in all_track_files if "_bg" not in file]
        return available_tracks

    def _set_track(self, track_image_path):
        self.track_boundaries = self._get_track_boundaries_from_image(track_image_path)
        self.middle_line = self._calculate_middle_line()
        self.progress_boundary = self._calculate_track_progress_boundary()
        background_image_path = Path(track_image_path.replace(".png", "_bg.png"))
        if background_image_path.exists():
            self.background_image = pg.image.load(background_image_path)
            scale_factor = self.ppu / 60
            self.background_image_scaled = pg.transform.scale(self.background_image, (int(self.background_image.get_width() * scale_factor), int(self.background_image.get_height() * scale_factor)))
        else:
            self.background_image = None
            self.background_image_scaled = None

    def _get_track_boundaries_from_image(self, image_path) -> dict:
        '''
        Using opencv to read the image and find the contours of the track boundaries.
        The boundaries are represented as list of points. ([x1, y1], [x2, y2], ...)
        '''
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Track image not found at {image_path}")
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        contours, _ = cv2.findContours(image.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        assert len(contours) == 2, "There should be exactly 2 contours in the image"

        # Transform and downsample the contours
        downsample_rate = 10
        inner_boundary = contours[0][:, 0, :][::downsample_rate]
        outer_boundary = contours[1][:, 0, :][::downsample_rate]

        # # Append last point to close the loop
        inner_boundary = np.append(inner_boundary, [inner_boundary[0]], axis=0)
        outer_boundary = np.append(outer_boundary, [outer_boundary[0]], axis=0)

        # Scale the contours to make suitable for the simulation
        inner_boundary, outer_boundary = inner_boundary / 60, outer_boundary / 60
        return {
            'inner': inner_boundary,
            'outer': outer_boundary
        }   

    def _calculate_middle_line(self):
        middle_line = []
        for point_inner in self.track_boundaries['inner']:
            point_outer = self._get_neares_point_from_boundary(Vector2(*point_inner), boundary='outer')
            middle_point = (Vector2(*point_inner) + Vector2(*point_outer)) / 2
            middle_line.append(middle_point)
        return middle_line

    def _init_lidar_sensor_vectors(self, sensor_angles):
        self.lidar_sensor_vectors = []
        for i in sensor_angles:
            angle = math.radians(i)
            self.lidar_sensor_vectors.append(Vector2(math.cos(angle), math.sin(angle)))
        self.lidar_sensor_distances = [0.0] * len(self.lidar_sensor_vectors)
        self.current_sensor_vectors = [] # stores the current sensor vectors rotated according to the car heading
        self.lidar_sensor_intercepts = [None] * len(self.lidar_sensor_vectors)
    
    def _calculate_track_progress_boundary(self):
        # calculate the index from the inner boundary that is closest to the start position
        inner_boundary = self.track_boundaries['inner']
        start_point = Vector2(self.position)
        start_point = self._get_neares_point_from_boundary(start_point, boundary='inner')
        start_index = np.where(np.all(inner_boundary == start_point, axis=1))[0][0]
        progress_boundary = np.roll(inner_boundary, -start_index-1, axis=0)
        progress_boundary = progress_boundary[::-1] # reverse the array
        return progress_boundary

    def update(self, dt):
        self._update_position(dt)
        self._update_lidar_data()
        self._update_track_collision()
        self._calculate_track_progress()
  
    def _calculate_track_progress(self):
        nearest_point = self._get_neares_point_from_boundary(self.position)
        index = np.where(np.all(self.progress_boundary == nearest_point, axis=1))[0][0]
        progress = index / len(self.progress_boundary)
        self.progress = progress
  
    def _update_position(self, dt):
        # Apply steering
        steering_angle = math.radians(self.steering * self.velocity_magnitude * 20)  # Adjust as needed

        self.heading = self.heading.rotate(steering_angle)

        # Apply drag
        drag_force = self.drag_coefficient * self.velocity_magnitude * self.velocity_magnitude
        self.velocity_magnitude -= drag_force * dt

        # Calculate velocity vector
        self.velocity = self.heading * self.velocity_magnitude

        # Apply drifting effect
        heading_perpendicular = self.heading.rotate(math.copysign(90, self.steering)).normalize()
        self.velocity += heading_perpendicular * self.drifting_coefficient * abs(self.steering * self.velocity_magnitude)
       
        self.velocity = self.heading * self.velocity_magnitude
        # Update position
        self.position += self.velocity * dt


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

        # Write controlls buffer
        self.controlls_buffer["left"] = left
        self.controlls_buffer["right"] = right
        
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

        # Write controlls buffer to draw the controlls in the UI
        self.controlls_buffer["up"] = up
        self.controlls_buffer["down"] = down
        self.controlls_buffer["boost"] = boost

    def _update_track_collision(self):
        '''Check if the car is colliding with the track boundaries'''
        # Simple approach to just check if any sensor is below a certain threshold
        sensors_min = min(self.lidar_sensor_distances)
        self.collision = sensors_min < 0.2

    def _update_camera_position(self, screen):
        self.camera_position = self.position - (Vector2(*screen.get_rect().center) / self.ppu)
        self.camera_position_smooth = self.camera_position_smooth * 0.92 + self.camera_position * 0.08

    def _update_lidar_data(self):
        '''Update the lidar sensor data by raycasting into the track boundaries'''
        self.current_sensor_vectors = []
        for i, sensor_vector in enumerate(self.lidar_sensor_vectors):
            sensor_vector = sensor_vector.rotate(-self.heading.angle_to(Vector2(1, 0)))
            self.current_sensor_vectors.append(sensor_vector)
            self.lidar_sensor_intercepts[i] = self._raycast(self.position, sensor_vector, self.lidar_max_distance)
            
            # set the distance to the sensor intercept
            if self.lidar_sensor_intercepts[i] is not None:
                self.lidar_sensor_distances[i] = (self.lidar_sensor_intercepts[i] - self.position).length()
            else:
                self.lidar_sensor_distances[i] = self.lidar_max_distance

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

    def _get_neares_point_from_boundary(self, position, boundary='inner'):
        # Get the nearest point from the inner boundary
        # Implemented by ChatGPT
        nearest_point = None
        nearest_distance = float('inf')
        for point in self.track_boundaries[boundary]:
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
        
    def draw(self, screen):

        self._update_camera_position(screen)
        self._draw_grid(screen)
        self._draw_track(screen)

        if self.graphics_mode == 'wireframe':
            self._draw_lidar(screen)
            self._draw_car_wireframe(screen)
            self._draw_tires(screen)

        elif self.graphics_mode == 'graphics':
            self._draw_car_sprite(screen)

        self._draw_ui(screen)

    def _draw_ui(self, screen):
        font = pg.font.Font(None, 24)
        text = font.render(f"Steering: {self.steering:.2f}", True, (255, 255, 255))
        screen.blit(text, (10, 10))
        text = font.render(f"Velocity: {self.velocity_magnitude:.2f}", True, (255, 255, 255))
        screen.blit(text, (10, 30))
        text = font.render(f"Progress: {self.progress:.2f}", True, (255, 255, 255))
        screen.blit(text, (10, 50))
        sensor_data = ", ".join([f"{d:.2f}" for d in self.lidar_sensor_distances])
        text = font.render(f"Sensors: {sensor_data}", True, (255, 255, 255))
        screen.blit(text, (10, 70))

        # Draw the avaialbel key controlls
        start_pos = (screen.get_width() - 200, screen.get_height() - 120)
        text = font.render(f"Switch Track: T", True, (255, 255, 255))
        screen.blit(text, (start_pos[0], start_pos[1]))
        text = font.render(f"Switch Graphics: G", True, (255, 255, 255))
        screen.blit(text, (start_pos[0], start_pos[1] + 20))
        text = font.render(f"Zoom: Mouse Wheel", True, (255, 255, 255))
        screen.blit(text, (start_pos[0], start_pos[1] + 40))
        text = font.render(f"Quit: ESC", True, (255, 255, 255))
        screen.blit(text, (start_pos[0], start_pos[1] + 60))


        # draw the controlls left right and boost as rectagles that change color when pressed.
        # the buttons should be alligned susch as the left right and up arrow key on a keyboard
        left_color = (255, 255, 255) if self.controlls_buffer["left"] else (100, 100, 100)
        right_color = (255, 255, 255) if self.controlls_buffer["right"] else (100, 100, 100)
        boost_color = (255, 255, 255) if self.controlls_buffer["boost"] else (100, 100, 100)
        width = 50
        top_left = (20, screen.get_height() - 80)
        pg.draw.rect(screen, left_color, pg.Rect(top_left, (width, width)))
        pg.draw.rect(screen, right_color, pg.Rect((top_left[0] + width*2, top_left[1]), (width, width)))
        pg.draw.rect(screen, boost_color, pg.Rect((top_left[0] + width, top_left[1] - width), (width, width)))


                                                  
    def _draw_lidar(self, screen: pg.Surface):
        for i, sensor_vector in enumerate(self.current_sensor_vectors):
            sensor_value = self.lidar_sensor_distances[i]
            sensor_color = (0, 255,0)
            start_pos = (self.position - self.camera_position_smooth) * self.ppu
            end_pos = (self.position + sensor_vector * self.lidar_sensor_distances[i] - self.camera_position_smooth) * self.ppu
            pg.draw.line(screen, sensor_color, start_pos, end_pos, 1)

            
    def _draw_track(self, screen):
        draw_background = self.graphics_mode == 'graphics'
        draw_boundary = self.graphics_mode == 'wireframe'

        if draw_background and self.background_image is not None:
            top_left = -self.camera_position_smooth * self.ppu
            screen.blit(self.background_image_scaled, top_left)

        if not draw_boundary:
            return
        
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
                pg.draw.line(screen, (255, 30, 30), p1, p2, 2)

        #draw midle line
        for i in range(len(self.middle_line) - 1):
            p1 = self.middle_line[i]
            p2 = self.middle_line[i + 1]
            p1 = (p1 - self.camera_position_smooth) * self.ppu
            p2 = (p2 - self.camera_position_smooth) * self.ppu
            pg.draw.line(screen, (50, 50, 50), p1, p2, 2)


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


    def _draw_car_sprite(self, screen):
        sprite = pg.image.load("assets/car1.png")
        sprite = pg.transform.scale(sprite, (int(self.length * self.ppu * 2), int(self.width * self.ppu * 2)))
        sprite = pg.transform.rotate(sprite, self.heading.angle_to(Vector2(-1, 0)))
        sprite_rect = sprite.get_rect()
        sprite_rect.center = (self.position - self.camera_position_smooth) * self.ppu
        screen.blit(sprite, sprite_rect)

    def _draw_car_wireframe(self, screen):
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


    
class MiniF1RLEnv(gymnasium.Env):

    def __init__(self):
        # Initialize pygame stuff
        self.render_mode = None
        self.screen: pg.Surface|None = None
        self.clock = pg.time.Clock()
        # Initialize car model
        self.car_model = CarModel(*CAR_START_POSITION)
        # Initialize environment
        # Actions: 0 = nothing, 1 = left, 2 = right, 3 = boost
        self.action_space = spaces.Discrete(4) 
        # Observation: 3 lidar sensors and the steering angle
        self.observation_space = spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32)
        self.reward = 0
        self.prev_reward = 0
        self.last_step = datetime.now()

        self.dt = 1/30 # Fixed timestep

    def step(self, action):

        # Update car model
        if action is not None:
            match action:
                case 0:  # Nothing
                    self.car_model.update_velocity(False, False, False, self.dt)
                    self.car_model.update_steering(False, False, self.dt)
                case 1:  # Left
                    self.car_model.update_velocity(True, False, False, self.dt)
                    self.car_model.update_steering(True, False, self.dt)
                case 2:  # Right
                    self.car_model.update_velocity(True, False, False, self.dt)
                    self.car_model.update_steering(False, True, self.dt)
                case 3:  # Boost
                    self.car_model.update_velocity(True, False, True, self.dt)
                    self.car_model.update_steering(False, False, self.dt)
                case _:
                    raise ValueError(f"Invalid action {action}")
            
        terminate = self.car_model.get_termination()
        truncate = self.car_model.get_truncate()
        observation = self.car_model.get_observation()
        step_reward = self.car_model.get_reward()
        if action is not None:
            self.reward -= 0.1 

        self.car_model.update(self.dt)
        return observation, step_reward, terminate, truncate, {}
        
    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.reward = 0
        self.prev_reward = 0

        self.car_model.reset()

        return self.step(None)[0], None

    def render(self, mode='human'):
        if self.screen is None:
            pg.init()
            pg.display.init()
            self.screen = pg.display.set_mode((800, 600), pg.RESIZABLE)
            pg.display.set_caption("MiniF1RL")


        self.screen.fill((0, 0, 0))
        self.car_model.draw(self.screen)
        pg.display.flip()

    def get_reward_weights(self):
        return {
            "W1_progress": self.car_model.W1_progress,
            "W2_speed": self.car_model.W2_speed,
            "W3_finish": self.car_model.W3_finish,
            "W4_collision": self.car_model.W4_collision
        }

    def handle_pg_events(self, human_control=False) -> bool:
        '''
        Handle pygame events and controlls
        Returns False if the game should be terminated
        '''

        if not hasattr(self, "keyctrl")\
              and human_control:
            self.keyctrl = {"up": False, 
                            "down": False,
                            "left": False, 
                            "right": False, 
                            "boost": False}
        events = pg.event.get()
        # Event handling
        done = False
        for event in events:
            if event.type == pg.QUIT:
                done = True
                continue
            
            # Handle key bindings
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    done = True
                    continue
                elif event.key == pg.K_g:
                    self.car_model.switch_graphics_mode()
                elif event.key == pg.K_t:
                    self.car_model.switch_track()
                elif event.key == pg.K_r:
                    self.reset()
                
            # Zoom in and out using the mouse wheel
            elif event.type == pg.MOUSEWHEEL:
                if event.y > 0:
                    self.car_model.zoom(4)
                elif event.y < 0:
                    self.car_model.zoom(-4)

        if not human_control:
            return done

        for event in events:
        
            # Handle key controlls for manual play
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_w:
                    self.keyctrl["up"] = True
                    self.keyctrl["boost"] = True
                elif event.key == pg.K_s:
                    self.keyctrl["down"] = True
                elif event.key == pg.K_a:
                    self.keyctrl["left"] = True
                elif event.key == pg.K_d:
                    self.keyctrl["right"] = True
               
   
            elif event.type == pg.KEYUP:
                if event.key == pg.K_w:
                    self.keyctrl["up"] = False
                    self.keyctrl["boost"] = False
                elif event.key == pg.K_s:
                    self.keyctrl["down"] = False
                elif event.key == pg.K_a:
                    self.keyctrl["left"] = False
                elif event.key == pg.K_d:
                    self.keyctrl["right"] = False
     
        self.car_model.update_velocity(self.keyctrl["up"], self.keyctrl["down"], self.keyctrl["boost"], self.dt)
        self.car_model.update_steering(self.keyctrl["left"], self.keyctrl["right"], self.dt)

        return done

    def close(self):
        pg.display.quit()
        pg.quit()
        self.screen = None
        

    def seed(self, seed=None):
        pass

if __name__ == '__main__':
    env = MiniF1RLEnv()

    pg.init()

    done = False

 
    last_frame = datetime.now()
    while not done:

        # Limit framerate for manual controlls
        if (datetime.now() - last_frame).total_seconds() < 1/100:
            continue
        last_frame = datetime.now()

        exit = env.handle_pg_events(human_control=True)

        observation, step_reward, terminate, info, idk = env.step(None)
        env.render()

        if terminate:
            env.reset()

        if exit:
            break

    env.close()
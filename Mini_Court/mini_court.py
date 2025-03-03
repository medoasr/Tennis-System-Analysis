import cv2
from cv2 import cvtColor

import constants
import numpy as np
from utils import  convert_meters_to_pixels,convert_pixel_distance_to_meters
class MiniCourt:
    def __init__(self,frame):
        self.drawing_rectangle_w=250
        self.drawing_rectangle_h=450
        self.buffer=50
        self.padding=20
        self.set_canvas_position(frame)
        self.mini_courtPosition()
        self.set_court_drawing_key_points()
        self.set_court_lines()

    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6, 7),
            (1, 3),

            (0, 1),
            (8, 9),
            (10, 11),
            (10, 11),
            (2, 3)
        ]
    def set_canvas_position(self,frame):
        frame=frame.copy()
        self.xend=frame.shape[1]-self.buffer
        self.yend=self.buffer+self.drawing_rectangle_h
        self.ystart=self.yend-self.drawing_rectangle_h
        self.xstart=self.xend-self.drawing_rectangle_w
    def mini_courtPosition(self):
        self.court_start_x=self.xstart+self.padding
        self.court_start_y=self.ystart+self.padding
        self.court_end_x = self.xstart - self.padding
        self.court_end_y = self.ystart - self.padding
        self.court_width=self.court_end_x-self.court_start_x

    def convert_meters_pixels(self,meters):
        return convert_meters_to_pixels(meters,constants.DOUBLE_LINE_WIDTH,self.court_width)
    def set_court_drawing_key_points(self):
            drawing_key_points = [0] * 28

            # point 0
            drawing_key_points[0], drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
            # point 1
            drawing_key_points[2], drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
            # point 2
            drawing_key_points[4] = int(self.court_start_x)
            drawing_key_points[5] = self.court_start_y + self.convert_meters_pixels(
                constants.HALF_COURT_LINE_HEIGHT * 2)
            # point 3
            drawing_key_points[6] = drawing_key_points[0] + self.court_width
            drawing_key_points[7] = drawing_key_points[5]
            # #point 4
            drawing_key_points[8] = drawing_key_points[0] + self.convert_meters_pixels(
                constants.DOUBLE_ALLY_DIFFERENCE)
            drawing_key_points[9] = drawing_key_points[1]
            # #point 5
            drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_pixels(
                constants.DOUBLE_ALLY_DIFFERENCE)
            drawing_key_points[11] = drawing_key_points[5]
            # #point 6
            drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_pixels(
                constants.DOUBLE_ALLY_DIFFERENCE)
            drawing_key_points[13] = drawing_key_points[3]
            # #point 7
            drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_pixels(
                constants.DOUBLE_ALLY_DIFFERENCE)
            drawing_key_points[15] = drawing_key_points[7]
            # #point 8
            drawing_key_points[16] = drawing_key_points[8]
            drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_pixels(
                constants.NO_MANS_LAND_HEIGHT)
            # # #point 9
            drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_pixels(constants.SINGLE_LINE_WIDTH)
            drawing_key_points[19] = drawing_key_points[17]
            # #point 10
            drawing_key_points[20] = drawing_key_points[10]
            drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_pixels(
                constants.NO_MANS_LAND_HEIGHT)
            # # #point 11
            drawing_key_points[22] = drawing_key_points[20] + self.convert_meters_pixels(constants.SINGLE_LINE_WIDTH)
            drawing_key_points[23] = drawing_key_points[21]
            # # #point 12
            drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18]) / 2)
            drawing_key_points[25] = drawing_key_points[17]
            # # #point 13
            drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22]) / 2)
            drawing_key_points[27] = drawing_key_points[21]

            self.drawing_key_points = drawing_key_points
    def draw_background_rectangle(self,frame):
        shapes=np.zeros_like(frame,np.uint8)
        cv2.rectangle(shapes,(self.xstart,self.ystart),(self.xend,self.yend),(255,255,255),-1)
        out=frame.copy()
        alpha=.5
        mask=shapes.astype(bool)
        out[mask]=cv2.addWeighted(frame,alpha,shapes,1-alpha,0)[mask]
        return out

    def draw_mini_court(self, frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            output_frames.append(frame)
        return output_frames


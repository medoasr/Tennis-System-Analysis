def convert_pixel_distance_to_meters(pixel_distance,refernce_h_pixel,refrence_h_meter):

    return (pixel_distance*refrence_h_meter)/refernce_h_pixel


def convert_meters_to_pixels(meters,refrenceinmeters,refrenceinpixels):
    return (meters*refrenceinpixels)/refrenceinmeters
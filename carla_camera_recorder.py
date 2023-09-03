import argparse
import sys

import carla
import os
import cv2

from carla import Vehicle, ActorBlueprint
from carla import World, Client, WeatherParameters, ColorConverter
from data_classes import DataWeatherParameters

import json_helper
from carla_api_helper import CarlaAPIHelper
from carla_recording_generator import CarlaDataGenerator
from json_helper import JSONHelper
import time

import math  # ADDED
import ImageLabelGenerator2
import ImageLabelGenerator3

import csv


class CarlaCameraRecorder:

    def __init__(self, carla_client: Client):
        self.ego_vehicle = None
        self.client = carla_client
        self.world: World = carla_client.get_world()
        self.map = self.world.get_map()
        # self.world.set_weather(carla.WeatherParameters.ClearSunset)  # ADDED

    def record_camera_in_simulation_run(self, seed: int, vehicle_id: int, width: int, height: int, begin_at: float,
                                        end_at: float) -> None:
        """
        Record the camera in the given simulation run for the given vehicle_id and save it as an mp4 file
        @param seed: The seed from which the recording should be recorded
        @param vehicle_id: The id of the vehicle that should be recorded
        """
        file_path = JSONHelper.get_path_from_seed(seed=seed, recording=True)
        print("Evaluate recorder data at path", file_path)
        # Unzip recorder file at path
        JSONHelper.extract_from_zip(file_path)
        # Log file path
        log_data_path = str(file_path).replace(".zip", ".log")
        # Check data from carla recording for validity
        info = self.client.show_recorder_file_info(log_data_path, True)
        if info == "File is not a CARLA recorder\n":
            print("The file at path", file_path, "is not a CARLA recorder")
            return

        if info.__contains__("not found"):
            print("The file at path", file_path, "cannot be found.")
            return

        # Get count of all ticks in the recorded file using the recorder_file_info and split
        replay_tick_count = int(info.split("Frames: ")[1].split("Duration")[0])

        if end_at == sys.maxsize:
            end_at = replay_tick_count * CarlaDataGenerator.SIMULATOR_FIXED_TICK_DELTA
        CarlaCameraRecorder.END_AT = end_at

        image_save_folder = CarlaCameraRecorder.get_image_save_folder(seed, vehicle_id, begin_at=begin_at,
                                                                      end_at=CarlaCameraRecorder.END_AT)
        if os.path.exists(image_save_folder):
            print(f"The files were already recorded at {image_save_folder}")
            return

        # Get map name of recording
        map_name = info.split("Map: ")[1].split("\nDate")[0]
        print(map_name)

        new_weather = carla.WeatherParameters.Default
        # Load map from recording
        self.client.load_world(map_name)

        #Load Weather from recording log
        weather_path = JSONHelper.get_weather_from_seed(seed=seed, recording=True)
        print("Wetter-Pfad: ", weather_path)
        JSONHelper.extract_from_zip(weather_path)
        self.world.set_weather(self.from_dict_to_wp(json_helper.load_weather(str(weather_path).replace(".zip", ".json"))))
        print(self.from_dict_to_wp(json_helper.load_weather(str(weather_path).replace(".zip", ".json"))))
        #print(carla.WeatherParameters.ClearNight)
        #self.world.set_weather(carla.WeatherParameters.ClearNight)


        # Get world for later use
        world: World = self.client.get_world()

        # Set synchronous mode settings
        new_settings = world.get_settings()
        new_settings.synchronous_mode = True
        new_settings.fixed_delta_seconds = CarlaDataGenerator.SIMULATOR_FIXED_TICK_DELTA
        world.apply_settings(new_settings)

        # add degree for rotating TrafficLights
        blueprint_library = world.get_blueprint_library()
        trafficlight_rotation = ""
        trafficlight_location = ""
        car = False
        if car:
            blueprint_library = blueprint_library.filter('vehicle.nissan.micra')
            print(blueprint_library)

            trafficlight_rotation = carla.Rotation(yaw=args.yaw)
            trafficlight_location = carla.Location(77.6919788, 4.83, 0.10324219)
            #trafficlight_location = carla.Location(102.11, 4.83, 0.10324219)
            trafficlight_transform = carla.Transform(trafficlight_location, trafficlight_rotation)


            #wurde zum drehen der Ampel verwendet

            #print(nissan_cars)
            #location = carla.Location(77.6919788, 4.83, 0.10324219)
            ##location = carla.Location(100.53, 4.83, 0.25)
            #location = carla.Location(77.48, 4, 1)
            #for car in nissan_cars:
                #print(carla.Actor.get_transform(bp))
            #for bp in trafficlight_bp:
                #print(bp.attribute)



        # Initialize necessary helper classes
        api_helper = CarlaAPIHelper(client, world)

        # Start replay of simulation
        api_helper.start_replaying(log_data_path)
        # A tick is necessary for the server to process the replay_file command
        world.tick()

        print("Start with simulation replay")

        vehicles = []

        while len(vehicles) == 0:
            vehicles = api_helper.get_vehicles()
            world.tick()

        print(*(map(lambda c: c.id, vehicles)))  # ADDED

        # Get the ego vehicle from the given vehicle id
        ego_vehicle: Vehicle = list(filter(lambda v: v.id == vehicle_id, vehicles))[0]

        # --------------
        # Spawn attached instance_segmentation camera
        # --------------
        cam_bp = None
        if args.rgb:  # ADDED
            cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        else:
            cam_bp = world.get_blueprint_library().find('sensor.camera.instance_segmentation')

        cam_bp.set_attribute("image_size_x", str(width))
        cam_bp.set_attribute("image_size_y", str(height))
        cam_bp.set_attribute("fov", str(105))


        # cam_bp.set_attribute("image_type", "Grayscale")

        cam_location = carla.Location(2, 0, 2) #(-2,0,3)
        cam_rotation = carla.Rotation(10, 0, 0)
        cam_transform = carla.Transform(cam_location, cam_rotation)
        ego_cam = world.spawn_actor(cam_bp, cam_transform, attach_to=ego_vehicle,
                                    attachment_type=carla.AttachmentType.Rigid)
        if args.rgb:
            ego_cam.listen(lambda image: CarlaCameraRecorder.save_image_data_semantic(world,image, seed, vehicle_id,
                                                                                      begin_at=begin_at, end_at=end_at))
        else:
            ego_cam.listen(lambda image: CarlaCameraRecorder.save_image_data_instance(world, image, seed, vehicle_id, begin_at=begin_at,
                                                                         end_at=end_at))
        spectator = world.get_spectator()

        # --------------
        # initialize obstacle detector (if activated)
        # --------------
        obs_bp = world.get_blueprint_library().find('sensor.other.obstacle')

        obs_bp.set_attribute("distance", str(50))
        obs_bp.set_attribute("hit_radius", str(50))
        obs_bp.set_attribute("only_dynamics", str(False))

        obs_location = carla.Location(0, 0, 0)
        obs_rotation = carla.Rotation(0, 0, 0)
        obs_transform = carla.Transform(obs_location, obs_rotation)

        obs_detector = 0


        def obstacle_detection_callback(event):
            print(f"Obstacle Detector: {event}")


        if args.obstacledetector:
            obs_detector = world.spawn_actor(obs_bp, obs_transform, attach_to=ego_cam)
            obs_detector.listen(lambda data: obstacle_detection_callback(data))



        # spawns rotated fake trafficlight
        if car:
            actor = world.spawn_actor(blueprint_library[0], carla.Transform(trafficlight_location, trafficlight_rotation))

        # Tick the world for each frame in the replay
        for tick in range(1, replay_tick_count):
            # Advance simulation by one tick
            world.tick()
            current_tick = CarlaDataGenerator.SIMULATOR_FIXED_TICK_DELTA * tick
            if current_tick < begin_at:
                print(f"Current tick {current_tick} is not within [{begin_at}, {end_at}]")
                continue
            if current_tick > end_at:
                print(f"Current tick {current_tick} is not within [{begin_at}, {end_at}]")
                break
            transform = ego_cam.get_transform()
            spectator.set_transform(carla.Transform(transform.location, transform.rotation))
            print(f"Tick {tick} of {replay_tick_count}. Simulation Tick: {current_tick}")
            #print(world.get_actors())
            while CarlaCameraRecorder.COUNTER < tick:
                x = ""

        client.reload_world()
        JSONHelper.delete_file(log_data_path)

    COUNTER: int = 0
    CURRENTLY_SAVING_IMAGE = False
    END_AT = 0.0

    @staticmethod
    def get_video_prefix(seed: int, vehicle_id: int, begin_at: float, end_at: float) -> os.path:
        if args.rgb:
            return f"seed_{seed}-vehicle_{vehicle_id}_range[{float(begin_at)}, {end_at}]" #ADDED
        else:
            return f"INSTANCE_seed_{seed}-vehicle_{vehicle_id}_range[{float(begin_at)}, {end_at}]"  # ADDED

    @staticmethod
    def get_image_save_folder(seed: int, vehicle_id: int, begin_at: float, end_at: float) -> os.path:
        folder_path = CarlaCameraRecorder.get_video_prefix(seed, vehicle_id, begin_at=begin_at, end_at=end_at)
        recording_folder = JSONHelper.get_experiment_data_folder()
        return os.path.join(recording_folder, JSONHelper.VIDEO_IMAGE_FOLDER, folder_path)

    @staticmethod
    def get_video_save_folder() -> os.path:
        recording_folder = JSONHelper.get_experiment_data_folder()
        return os.path.join(recording_folder, JSONHelper.VIDEO_FOLDER)

    @staticmethod
    def save_image_data_instance(world, image, seed: int, vehicle_id: int, begin_at: float, end_at: float):
        CarlaCameraRecorder.COUNTER += 1
        current_tick = CarlaDataGenerator.SIMULATOR_FIXED_TICK_DELTA * CarlaCameraRecorder.COUNTER
        if begin_at <= current_tick <= end_at and CarlaCameraRecorder.COUNTER % 10 == 0 and (CarlaCameraRecorder.COUNTER / 5) > args.start_at:
            image_name = f"%.6d_INSTANCE_seed{args.seed}.png" % (CarlaCameraRecorder.COUNTER / 5)
            label_name = f"%.6d_seed{args.seed}.png" % (CarlaCameraRecorder.COUNTER / 5)

            #image_name = f"%.6d_INSTANCE.png" % CarlaCameraRecorder.COUNTER
            image_save_folder = CarlaCameraRecorder.get_image_save_folder(seed, vehicle_id, begin_at=begin_at,
                                                                          end_at=end_at)
            recording_path = os.path.join(image_save_folder, image_name)
            print(f"Save image {image_name}")
            image.save_to_disk(recording_path)

            actorlist = world.get_actors()

            tl_list = actorlist.filter('traffic.traffic_light')
            print(tl_list)


            sensor_actor = actorlist.filter('sensor.camera.instance_segmentation')[0]

            label = ImageLabelGenerator3.checkImage(tl_list, sensor_actor, recording_path)

            os.remove(recording_path)


            with open(image_save_folder+f"/labels_{args.seed}.csv",'a', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow([label_name, label])





    @staticmethod
    def save_image_data_semantic(world: World, image, seed: int, vehicle_id: int, begin_at: float, end_at: float):  # ADDED
        CarlaCameraRecorder.COUNTER += 1
        current_tick = CarlaDataGenerator.SIMULATOR_FIXED_TICK_DELTA * CarlaCameraRecorder.COUNTER
        if begin_at <= current_tick <= end_at:

            actorlist = world.get_actors()
            tl_list = actorlist.filter('traffic.traffic_light')
            tl_id_list = [actor.id for actor in tl_list]

            # print(tl_id_list)

            camera_location = actorlist.filter('sensor.camera.rgb')[0].get_location()

            # calculate angle between car and trafficlight (old?)
            car = False
            if car:
                #wurde zur temporären Winkelbestimmung zwischen Ampel und Kamera verwendet
                trafficlight_location = carla.Location(77.6919788, 4.83, 0.10324219) #1st normal trafficlight
                #trafficlight_location = carla.Location(102.43, 4.83, 0.10324219)  # 1st normal trafficlight
                #trafficlight_location = carla.Location(85.73, 45.5680212, 0.10324219) #2st normal trafficlight

                #trafficlight_location = carla.Location(13.70279815, -213.7444715, 0.10324219) #AmericanLights_0
                #trafficlight_location = carla.Location(10.70224365, -213.74447173, 0.10324219)  # AmericanLights_1
                #trafficlight_location = carla.Location(7.20242797, -213.74447181, 0.10324219)  # AmericanLights_2

                distance_x = abs(trafficlight_location.x - camera_location.x)
                distance_y = abs(trafficlight_location.y - camera_location.y)

                degree = math.atan(distance_y/distance_x) * (180 / math.pi) #Rad to degree
                #degree = math.atan(distance_x / distance_y) * (180 / math.pi)  # Rad to degree
                degree = round(degree,2)

            #image_name = f"%.6d_SEMANTIC_degree_{degree}yaw{args.yaw - 90}.png" % CarlaCameraRecorder.COUNTER

            test_distance_tl = carla.Location(323.410097, 202.030078, 0.10326172)

            distance = test_distance_tl.distance(camera_location)
            degree = "X"

            image_name = f"%.6d_degree_X_yaw_{args.yaw}.png" % CarlaCameraRecorder.COUNTER
            image_save_folder = CarlaCameraRecorder.get_image_save_folder(seed, vehicle_id, begin_at=begin_at,
                                                                          end_at=end_at)
            recording_path = os.path.join(image_save_folder, image_name)
            print(f"Save image {image_name}")
            #image.save_to_disk(recording_path, carla.ColorConverter.CityScapesPalette)
            image.save_to_disk(recording_path)

    @staticmethod
    def save_video(seed: int, vehicle_id: int, begin_at: float, end_at: float, semantic = False):
        image_folder = CarlaCameraRecorder.get_image_save_folder(seed, vehicle_id, begin_at=begin_at, end_at=end_at)

        video_folder = CarlaCameraRecorder.get_video_save_folder()
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)
        if semantic:
            video_name = f"{CarlaCameraRecorder.get_video_prefix(seed, vehicle_id, begin_at, end_at)}.mp4"
        else:
            video_name = f"INSTANCE_{CarlaCameraRecorder.get_video_prefix(seed, vehicle_id, begin_at, end_at)}.mp4"
        video_path = os.path.join(video_folder, video_name)

        if os.path.exists(video_path):
            print(f"The video was already produced at {video_path}")
            return

        images_in_folder = os.listdir(image_folder)
        if images_in_folder.__sizeof__() == 0:
            print("There are no images to save as video")
            return

        images = [img for img in images_in_folder if img.endswith(".png")]

        images = images[0:-1]

        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video = cv2.VideoWriter(video_path, fourcc, 20, (width, height))
        print(f"Save video to {video_path}")

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()


    @staticmethod
    def from_dict_to_wp(dwp: DataWeatherParameters) -> WeatherParameters:  # ADDED
        type = dwp.type
        cloudiness = dwp.cloudiness
        precipitation = dwp.precipitation
        precipitation_deposits = dwp.precipitation_deposits
        wind_intensity = dwp.wind_intensity
        sun_azimuth_angle = dwp.sun_azimuth_angle
        sun_altitude_angle = dwp.sun_altitude_angle
        fog_density = dwp.fog_density
        fog_distance = dwp.fog_distance
        wetness = dwp.wetness
        fog_falloff = dwp.fog_falloff
        scattering_intensity = dwp.scattering_intensity
        mie_scattering_scale = dwp.mie_scattering_scale
        rayleigh_scattering_scale = dwp.rayleigh_scattering_scale

        weatherparam= WeatherParameters(cloudiness, precipitation, precipitation_deposits, wind_intensity,
                                 sun_azimuth_angle,sun_altitude_angle,
                                 fog_density, fog_distance, wetness, fog_falloff, scattering_intensity,
                                 mie_scattering_scale,rayleigh_scattering_scale)


        return weatherparam


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-s', '--seed',
        metavar='S',
        type=str,
        default=50,
        help='Set seed for which the recording should be loaded')
    argparser.add_argument(
        '-v', '--vehicle-id',
        metavar='V',
        type=int,
        default=500, #111, #seed2: 290, seed1: 135 vorher 290
        help='For which vehicle id should the camera be recorded?')
    argparser.add_argument(
        '-x', '--width',
        metavar='V',
        type=int,
        default=640,
        help='Width of the resulting video')
    argparser.add_argument(
        '-y', '--height',
        metavar='V',
        type=int,
        default=480,
        help='Height of the resulting video')
    argparser.add_argument(
        '-b', '--begin_at',
        metavar='B',
        type=float,
        default=0, #164.15,#59.3, #49.3, #0.0,
        help='Tick at which the video should start')
    argparser.add_argument(
        '-e', '--end_at',
        metavar='E',
        type=float,
        default=sys.maxsize, #207.2,#86.5,#61.1, #sys.maxsize,
        help='Tick at which the video should end')
    argparser.add_argument(  # ADDED
        '--rgb',
        metavar='RGB',
        default=False,
        type=bool,
        action='store',
        help='True turns on RGB Frame Sampling')
    argparser.add_argument(  # ADDED
        '--yaw', '-r',
        metavar='YAW',
        default=90,
        type=int,
        action='store',
        help='Amount of degrees the test-TrafficLight should rotate (y-axis)')
    argparser.add_argument(  # ADDED
        '--obstacledetector',
        metavar='obstacledetector',
        default=False,
        type=bool,
        action='store',
        help='Turns off/on the obstacledetector and returns obstacles around car')
    argparser.add_argument(  # ADDED
        '--start_at',
        metavar='start',
        default=2260,
        type=int,
        action='store',
        help='defines at which frame the sampling should start')

    args = argparser.parse_args()

    seed = args.seed
    vehicle_id = args.vehicle_id
    begin_at = args.begin_at
    end_at = args.end_at

    video_width = args.width
    video_height = args.height

    print("Proceed with the following arguments:")
    print(f"Seed: {seed}, Vehicle Id: {vehicle_id}, Tick Range: [{begin_at}, {end_at}] ")
    print(f"Video Width: {video_width}, Video Height: {video_height}")

    print("Connect to Carla")
    print(args.yaw)
    # Find carla simulator at localhost on port 2000
    client = carla.Client('localhost', 2000)

    # Try to connect for 10 seconds. Fail if not successful
    client.set_timeout(60.0)
    recorder = CarlaCameraRecorder(carla_client=client)
    print("Connected to carla")
    try:
        recorder.record_camera_in_simulation_run(seed=seed, vehicle_id=vehicle_id, width=video_width,
                                                 height=video_height,
                                                 begin_at=begin_at, end_at=end_at)
        print("Done with monitoring the recording")
    finally:
        print("Convert images to video")
        CarlaCameraRecorder.save_video(seed=seed, vehicle_id=vehicle_id, begin_at=begin_at,
                                       end_at=CarlaCameraRecorder.END_AT, semantic=False)

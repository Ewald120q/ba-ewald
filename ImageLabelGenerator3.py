import pandas
import numpy as np
import os
import glob
import matplotlib.image as imread
import sys
import os.path
import math
from numba import jit

max_angle = 70 # max_angle between z-rotation of sensor and trafficlight in degree    #???
max_distance = 63 # max_eucl. distance between sensor and trafficlight in meters      #???
colors = []

# checkID-method by https://github.com/carla-simulator/carla/discussions/5047 from gogokotsev00
def checkID(actor_id, G, B):
    boundary_value = 65536
    if actor_id > boundary_value:
        actor_id -= math.floor(actor_id / boundary_value) * boundary_value
    if (B<<8) + G == actor_id:
            #print ("Ampel-ID wurde erkannt")
            return True
    #print("Diese ID ist keine Ampel")
    return False

def getVector3D(actor, isTrafficlight):
    rotation = actor.get_transform().rotation
    roll = np.deg2rad(rotation.roll)
    pitch = np.deg2rad(rotation.pitch)
    yaw = 0
    if isTrafficlight: # tl and cars are not equally oriented in carla. Without correction, vectors wouldn't be comparable
        yaw = np.deg2rad(rotation.yaw-90)
        print(f"Ampel: {rotation}")
    else:
        yaw = np.deg2rad(rotation.yaw)
        print(f"Auto: {rotation}")
    rotation_matrix = np.array([[np.cos(yaw) * np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll)],
                                [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll)+np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.sin(pitch)*np.cos(roll)-np.cos(yaw)*np.sin(roll)],
                                [-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)]])



    x = np.array([[1],[1],[1]])

    #print(f"rotation_vector: {np.matmul(rotation_matrix, x)}")
    return np.matmul(rotation_matrix, x)


def smooth(x):
    if x > 1:
        return [[1]]
    if x < -1:
        return [[-1]]
    return x

def getVector2D(actor, isTrafficlight):
    rotation = actor.get_transform().rotation
    alpha = np.deg2rad(rotation.roll) #roll
    pitch = 0
    beta = 0 #yaw
    if isTrafficlight: # tl and cars are not equally oriented in carla. Without correction, vectors wouldn't be comparable
        beta = np.deg2rad(rotation.yaw-90)
        print(f"Ampel: {rotation}")
    else:
        beta = np.deg2rad(rotation.yaw)
        print(f"Auto: {rotation}")
    rotation_matrix = np.array([[np.cos(alpha) * np.cos(beta) - np.sin(alpha) * np.sin(beta), -np.cos(alpha) * np.sin(beta)- np.sin(alpha) * np.cos(beta)],
                                [np.cos(alpha) * np.sin(beta) + np.sin(alpha) * np.cos(beta), np.cos(alpha) * np.cos(beta)- np.sin(alpha) * np.sin(beta)]])



    x = np.array([[1],[1]])

    #print(f"rotation_vector: {np.matmul(rotation_matrix, x)}")
    return np.matmul(rotation_matrix, x)





def getAngle(trafficlight_rotation, sensor_rotation):
    tl_vector = getVector2D(trafficlight_rotation, isTrafficlight = True)
    #print(f"tl_vector:\n {tl_vector}")
    sensor_vector = getVector2D(sensor_rotation, isTrafficlight = False)
    #print(f"sensor_vector:\n {sensor_vector}")
    #print("\n")
    # https://stackoverflow.com/questions/9171158/how-do-you-get-the-magnitude-of-a-vector-in-numpy
    # rundet manchmal komisch, sodass beim bruch etwas ganz leicht über 1 rauskommt. deshalb min(x,1)
    x = min(np.dot(tl_vector.T, sensor_vector) / (np.sqrt(np.dot(tl_vector.T, tl_vector)) * np.sqrt(np.dot(sensor_vector.T, sensor_vector))), 1)
    return np.rad2deg(np.arccos(smooth(x))) #kann passieren dass durch Rundungen x=1.00000...1 ist. Würde NaN ausgeben




def checkFeatures(trafficlight_actor, sensor_actor):

    #check for distance
    location_tl = trafficlight_actor.get_location()
    location_sensor = sensor_actor.get_location()
    distance = abs(location_sensor.distance(location_tl)) # sign not important, because trafficlight cannot be
    # behind the sensor, otherwise it wouldn't be on the image

    print(f"distance: {distance}")
    if distance > max_distance:
        print("Ampel zu weit weg")
        return False


    #check for angle
    angle = getAngle(trafficlight_actor, sensor_actor)

    print(f"angle: {angle}")
    if angle > max_angle:
        print("Winkel zu groß")
        return False

    return True

@jit(forceobj=True)
def checkImageLoop(used_trafficlight_actors, sensor_actor):
    for tl_actor in used_trafficlight_actors:
        if checkFeatures(tl_actor, sensor_actor):
            print("Es wurde eine Ampel gefunden! HURRA!")
            return 1

    print("Es wurde keine Ampel gefunden")
    return 0

def getUsedTrafficLightActors(colors, tl_list, trafficlight_ids):
    used_trafficlight_ids = []
    for color in colors:
        # print(f"trafficlight_ids: {trafficlight_ids}")
        # print(color[1])
        # print(color[2])

        trafficlight_id = list(filter(lambda x: checkID(x, int(color[1]), int(color[2])), trafficlight_ids))

        # print(f"trafficlight: {trafficlight_id}")
        used_trafficlight_ids.extend(trafficlight_id)

    used_trafficlight_actors = [tl_list.find(actorid) for actorid in used_trafficlight_ids]
    return used_trafficlight_actors

@jit(forceobj=True)
def loop(x,y,trafficlight_ids,tl_list, ins_img, sensor_actor):
    for i in range(len(y)):
        color = ins_img[x[i], y[i]]
        # print(f"color: {color}")
        # print(type(color))
        if not np.any(np.all(color == colors)):
            colors.append(color)
            used_trafficlight_actors = getUsedTrafficLightActors([color], tl_list, trafficlight_ids)
            currentTLrelevant = checkImageLoop(used_trafficlight_actors, sensor_actor)
            if currentTLrelevant == 1:
                return 1
    return 0


def checkImage(tl_list, sensor_actor, instance_path):


    semantic_path = instance_path.replace("INSTANCE", "SEMANTIC")
    #print("semantic_path: "+semantic_path)
    image = imread.imread(semantic_path)
    image2 = imread.imread(instance_path)


    sem_img = np.array(image)
    ins_img = np.array(image2)*255

    r_rightColor = 250/255
    g_rightColor = 170/255
    b_rightColor = 30/255


    x, y = (np.where((sem_img[:, :, 0] == r_rightColor) & (sem_img[:, :, 1] == g_rightColor) & (sem_img[:, :, 2] == b_rightColor))) #pos of every right_color pixel

    if len(y) == 0: #checks, if we have at least one rightColor Pixel on semantic image
        #print("leer")
        return 0

    else:
        #print("nicht leer")
        #print(f"X: {x}")
        #print(f"Y: {y}")
        trafficlight_ids = [actor.id for actor in tl_list]
        #print(f"len(y): {len(y)}")
        colors.clear()
        return loop(x,y,trafficlight_ids, tl_list, ins_img, sensor_actor)
                #print(color)
                #print(colors)
        #print(f"colors: {colors}")




        #print("used trafficlight actors: ")
        #print(used_trafficlight_actors)

        # now we got every actor-instance of all trafficlights on the image








if __name__ == '__main__':
    #read color that we want to search
    rightColor = imread.imread('right_color.jpg')[0][0]
    print("rightColor: " + str(rightColor))
    #get our images that we want to scan


# cite: https://github.com/carla-simulator/carla/discussions/5047 from gogokotsev00
import math
def checkID(actor_id, G, B):
    boundary_value = 65536
    if actor_id > boundary_value:
        actor_id -= math.floor(actor_id / boundary_value) * boundary_value
    if (B<<8) + G == actor_id:
            return True
    return False


if __name__ == '__main__':

    list = [90957, 90939, 90935, 90931, 90927, 90923, 90919, 90915, 90911, 90907, 90903, 90899, 90895, 90891, 90887, 90856, 90852, 90848, 90844, 90840, 90836, 90832, 90828, 90812, 90806, 90802, 90790, 90786, 90780, 90776, 90772, 90761, 90757, 90753, 90749, 90745]


    for element in list:
        print(checkID(element, 55,99))
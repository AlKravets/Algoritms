import numpy as np
import pprint
import json

np.random.seed(20)

# fields  = [[x_min,y_min],[x_max,y_max]]
fields = np.array([
    [0,0],
    [1,1]
])


def create_random_receivers(n):
    '''
    Возвращает np.array со случайными координатами приемников
    в области fields
    '''
    return np.random.rand(n,2)*(fields[1]-fields[0])+ fields[0]


def create_random_points(m):
    '''
    Возвращает np.array со случайными координатами в области fields
    (предположительно местоположением человека в конкретный момент времени)

    '''
    return np.random.rand(m,2)*(fields[1]-fields[0])+ fields[0]


def create_dictance(receivers, points):
    '''
    Создание массива дистанций с шумами
    '''
    #print(points.shape[0], receivers.shape[0])
    distance = np.zeros((points.shape[0], receivers.shape[0]))

    for i in range(points.shape[0]):
        for j in range(receivers.shape[0]):
            distance[i,j] = ((points[i][0]- receivers[j][0])**2 + (points[i][1]- receivers[j][1])**2)**0.5
    return deformation(distance)

    


    

def deformation (arr, noise=0.005, type = 'N'):
    '''
    возвращает массив с шумами +- noise распределенных нормально(type = 'N')
    равномерно (type = 'U')
    '''
    if type == 'N':
        return arr + np.random.randn(arr.shape[0], arr.shape[1])*noise - noise
    elif type == 'U':
        return arr + np.random.rand(arr.shape[0], arr.shape[1])*noise - noise
    else:
        print('Error')


def distance_2_points(a,b):
    return ((a[0]- b[0])**2 + (a[1]- b[1])**2)**0.5



def metod_1 (distance, points, receivers, zero_point = np.array([(fields[1][0] -fields[0][0])/2, (fields[1][1] -fields[0][1])/2]), eps = 0.005):
    '''
    Метод Гаусса-Ньютона
    '''
    result = []
    for i  in range(points.shape[0]):
        Jf = np.zeros((distance[i].shape[0],2))
        F = np.zeros(distance[i].shape[0])

        count = 0
        c_k = zero_point
        while count< 2000 and distance_2_points(points[i], c_k) >eps:

            for j in range(distance[i].shape[0]):
                F[j] = c_k[0]**2 + c_k[1]**2 - 2*receivers[j,0]*c_k[0] - 2*receivers[j,1]*c_k[1] + receivers[j][0]**2 + receivers[j][1]**2 - distance[i,j]**2
                Jf[j] = [ 2*c_k[0]-2*receivers[j,0], 2*c_k[1] - 2*receivers[j,1] ]
            
            
            c_k = c_k - np.dot( np.dot( np.linalg.inv( np.dot(np.transpose(Jf), Jf)) , np.transpose(Jf) ), F )
            count+=1


        result.append( {
            'real point': [points[i][0], points[i][1]],
            'found point': [c_k[0], c_k[1]],
            'distance': distance_2_points(points[i], c_k),
            'count': count,
            'metod': 'Gauss Newton'
        } )
    return result



def metod_2 (distance, points, receivers, zero_point = np.array([(fields[1][0] -fields[0][0])/2, (fields[1][1] -fields[0][1])/2]), eps = 0.005, lm = 1):
    '''
    Метод Ленинисберга-Маркуарта
    '''
    result = []
    for i  in range(points.shape[0]):
        Jf = np.zeros((distance[i].shape[0],2))
        F = np.zeros(distance[i].shape[0])

        count = 0
        c_k = zero_point
        while count< 2000 and distance_2_points(points[i], c_k) >eps:

            for j in range(distance[i].shape[0]):
                F[j] = c_k[0]**2 + c_k[1]**2 - 2*receivers[j,0]*c_k[0] - 2*receivers[j,1]*c_k[1] + receivers[j][0]**2 + receivers[j][1]**2 - distance[i,j]**2
                Jf[j] = [ 2*c_k[0]-2*receivers[j,0], 2*c_k[1] - 2*receivers[j,1] ]
            
            
            c_k = c_k - np.dot( np.dot( np.linalg.inv( np.dot(np.transpose(Jf), Jf)+ lm*np.eye(2) ) , np.transpose(Jf) ), F )
            count+=1


        result.append( {
            'real point': [points[i][0], points[i][1]],
            'found point': [c_k[0], c_k[1]],
            'distance': distance_2_points(points[i], c_k),
            'count': count,
            'metod': 'Levenberg Marquardt'
        } )
    return result



if __name__ == "__main__":
    #print(create_random_receivers(10))
    

    n = 5
    m = 10

    receivers = create_random_receivers(n)
    points = create_random_points(m)

    distance = create_dictance(receivers, points)

    result_1 = metod_1(distance, points, receivers)
    result_2 = metod_2(distance, points, receivers)
    result = result_1 + result_2


    comparison_list = []


    for r1 in result_1:
        for r2 in result_2:
            if r1['real point'] == r2['real point']:
                comparison_list.append({
                    'point': r1['real point'],
                    'dist {} - dist {}'.format(r1['metod'], r2['metod']): r1['distance']- r2['distance']
                })

    with open('log.txt', 'w') as f:
        json.dump(result,f, indent=4)

    pprint.pprint(comparison_list)
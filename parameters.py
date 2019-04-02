import os


def initialize(theta0=0, theta1=0):
    params_string = "{}\n{}".format(theta0, theta1)
    with open('params.txt', 'w+') as f:
        f.write(params_string)


def get_parameters():
    if not os.path.exists('params.txt'):
        initialize()
    with open('params.txt', 'r') as f:
        params = f.read().split("\n")
    return float(params[0]), float(params[1])


def delete():
    if os.path.exists('params.txt'):
        os.remove('params.txt')


if __name__ == '__main__':
    initialize(2, 3)
    print(get_parameters())
    delete()

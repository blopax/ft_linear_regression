import pandas as pd
import math
import argparse

import parameters
import predict
import plot


def get_df(filename):
    try:
        df = pd.read_csv(filename)
        return df
    except Exception as e:
        print("Training was not successful.")
        print(e)


def get_mean(serie):
    serie_sum = 0
    for item in serie:
        serie_sum += item
    return float(serie_sum / len(serie))


def get_std_deviation(serie, mean):
    variance = 0
    for item in serie:
        variance += (item - mean) ** 2
    variance /= len(serie)
    return float(math.sqrt(variance))


def normalize_serie(serie, mean, std_deviation):
    return (serie.astype(float) - mean) / std_deviation


def get_cost(df, theta0, theta1):
    cost_sum = 0
    for i in range(len(df)):
        diff = predict.predict(df['km'][i], theta0, theta1) - df['price'][i]
        cost_sum += diff ** 2
    return cost_sum / (2 * len(df))


def gradient_descent_step(theta0, theta1, learning_rate, df):
    sum0, sum1 = 0, 0
    for i in range(len(df)):
        diff = predict.predict(df['normalized_km'][i], theta0, theta1) - df['normalized_price'][i]
        sum0 += diff
        sum1 += (diff * df['normalized_km'][i])
    tmp0 = theta0 - learning_rate * float(sum0 / len(df))
    tmp1 = theta1 - learning_rate * float(sum1 / len(df))
    return tmp0, tmp1


def train_initialize_data(filename):
    df = get_df(filename)
    if df is None:
        return None, None
    norm_dic = {
        'price_mean': get_mean(df['price']),
        'km_mean': get_mean(df['km']),
    }
    norm_dic['price_std_deviation'] = get_std_deviation(df['price'], norm_dic['price_mean'])
    norm_dic['km_std_deviation'] = get_std_deviation(df['km'], norm_dic['km_mean'])
    if norm_dic['price_std_deviation'] == 0 or norm_dic['km_std_deviation'] == 0:
        return df, None

    df['normalized_price'] = normalize_serie(df['price'], norm_dic['price_mean'], norm_dic['price_std_deviation'])
    df['normalized_km'] = normalize_serie(df['km'], norm_dic['km_mean'], norm_dic['km_std_deviation'])

    return df, norm_dic


def train_normalized(df, norm_theta0, norm_theta1, norm_dic, args):
    iterations = args.iterations
    cost_list = []
    i = 0
    for i in range(iterations):
        norm_theta0, norm_theta1 = gradient_descent_step(norm_theta0, norm_theta1, args.learning_rate, df)
        theta0, theta1 = denormalize_thetas(norm_theta0, norm_theta1, norm_dic)
        cost_list.append(get_cost(df, theta0, theta1))
        if len(cost_list) > 2 and (cost_list[-2] - cost_list[-1]) / cost_list[-2] < args.stop:
            break
        if args.evolution and i % (iterations // 5) == 0:
            plot.lr_plot(theta0, theta1, True)
    print("Training stopped after {} iterations.".format(i))
    if args.show_cost:
        plot.cost_plot(cost_list)
    return norm_theta0, norm_theta1


def normalize_thetas(theta0, theta1, norm_dic):
    price_mean = norm_dic['price_mean']
    km_mean = norm_dic['km_mean']
    price_std = norm_dic['price_std_deviation']
    km_std = norm_dic['km_std_deviation']
    norm_theta0 = (theta0 - price_mean + theta1 * km_mean) / price_std
    norm_theta1 = km_std / price_std * theta1

    return norm_theta0, norm_theta1


def denormalize_thetas(norm_theta0, norm_theta1, norm_dic):
    price_mean = norm_dic['price_mean']
    km_mean = norm_dic['km_mean']
    price_std = norm_dic['price_std_deviation']
    km_std = norm_dic['km_std_deviation']

    theta0 = price_mean + price_std * (norm_theta0 - norm_theta1 * km_mean / km_std)
    theta1 = price_std / km_std * norm_theta1

    return theta0, theta1


def train(filename, args):
    df, norm_dic = train_initialize_data(filename)
    if df is None:
        print("No data.\n")
        return None, None
    if norm_dic is None:
        print("Standard deviation of mileage or price is null. No training performed.\n")
        return None, None
    if len(df) < 2:
        print("Not enough data points in the csv. No training performed.\n")
        return None, None
    theta0, theta1 = parameters.get_parameters()
    norm_theta0, norm_theta1 = normalize_thetas(theta0, theta1, norm_dic)
    norm_theta0, norm_theta1 = train_normalized(df, norm_theta0, norm_theta1, norm_dic, args)
    theta0, theta1 = denormalize_thetas(norm_theta0, norm_theta1, norm_dic)
    return theta0, theta1


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--reinitialize", action="store_true",
                        help="Reinitialize everything without training\n")
    parser.add_argument("-b", "--theta0", type=float, help="Set theta0\n")
    parser.add_argument("-a", "--theta1", type=float, help="Set theta1\n")
    parser.add_argument("-i", "--iterations", type=float, default=10000,
                        help="Choose the learning rate.\n")
    parser.add_argument("-s", "--stop", type=float, default=0.00001,
                        help="Stop when cost function evolution lower than.\n")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001,
                        help="Choose the learning rate.\n")
    parser.add_argument("-v", "--visualization", action="store_true", help="Show plot and fit line.\n")
    parser.add_argument("-e", "--evolution", action="store_true", help="Show how the plot evolve during learning.\n")
    parser.add_argument("-c", "--show_cost", action="store_true", help="Show cost function.\n")

    return parser.parse_args()


if __name__ == "__main__":
    try:
        options = get_args()
        filepath = 'data.csv'
        if options.reinitialize:
            parameters.delete()
            print("theta0 and theta1 were reinitialized.")
        else:
            if options.theta0:
                parameters.initialize(theta0=options.theta0)
            if options.theta1:
                parameters.initialize(theta1=options.theta1)
            t0, t1 = train(filepath, options)
            if t0 is not None and t1 is not None:
                parameters.initialize(t0, t1)
                print("theta0 = {}\ntheta1 = {}".format(t0, t1))
                if options.visualization:
                    plot.lr_plot()
    except Exception as error:
        print("An error occured: {}".format(error))

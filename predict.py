import argparse

import parameters
import plot


def predict(mileage, theta0=None, theta1=None):
    if not theta0 or not theta1:
        theta0, theta1 = parameters.get_parameters()
    prediction = theta0 + theta1 * mileage
    return prediction


def define_expected_km_for_budget(budget, theta0=None, theta1=None):
    if not theta0 or not theta1:
        theta0, theta1 = parameters.get_parameters()
    if theta1 == 0:
        expected_km = None
    else:
        expected_km = int((budget - theta0) / theta1)
    return expected_km


def predict_display(ask_budget=False, non_stop=True):
    keep_going = True
    while keep_going:
        keep_going = non_stop
        if not ask_budget:
            user_input = input("What is the mileage of your car? (press q to quit)\n")
        else:
            user_input = input("What is your budget? (press q to quit)\n")
        if user_input == 'q':
            keep_going = False
        elif len(set(user_input)) > 0 and set(user_input).issubset({'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}):
            if not ask_budget:
                price = predict(int(user_input))
                print("The estimated price of your car is: {} euros.".format(int(price)))
            else:
                km = define_expected_km_for_budget((int(user_input)))
                if km is None:
                    print("Error. Theta1 = 0. Try to train the algorithm.\n")
                else:
                    print("With this budget look for a car with a mileage of {} km.".format(km))
        else:
            print("Wrong input. It must be an integer\n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="Enable the user to keep puting input.\n")
    parser.add_argument("-b", "--budget", action="store_true",
                        help="Enable the user to enter his budget and tell him what mileage he should expect\n")
    parser.add_argument("-v", "--visualization", action="store_true", help="Show plot.\n")
    return parser.parse_args()


if __name__ == '__main__':
    options = get_args()
    predict_display(options.budget, options.interactive)
    if options.visualization:
        plot.lr_plot(show_line=False)

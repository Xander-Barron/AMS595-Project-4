import numpy as np
import sympy as sp
import pandas as pd
import time
import matplotlib.pyplot as plt

def taylor_approximation(func, start, end, degree, c):
    #use x the variable not the character
    x = sp.symbols('x')

    #derivatives up to the nth one
    derivatives = [func.diff(x, n) for n in range(degree + 1)]

    #l00 evenly spaced x-values
    x_vals = np.linspace(start, end, 100)
    #initialize array of zeroes
    approx_vals = np.zeros_like(x_vals, dtype=float)

    for n in range(degree + 1):
        #fixed around c
        nth_term = (derivatives[n].subs(x, c) / sp.factorial(n)) * (x_vals - c) ** n #nth term of the series
        approx_vals += np.array(nth_term, dtype=float) #add up the series

    return x_vals, approx_vals


def factorial_analysis(func, start, end, c, initial_degree, final_degree, degree_step):
    x = sp.symbols('x')
    #done for numeric evaluation
    f_lambdified = sp.lambdify(x, func, modules='numpy')
    x_vals = np.linspace(start, end, 100)
    #compute the true values at the above x points
    true_vals = f_lambdified(x_vals)

    #initialize list
    records = []

    #loop through different taylor series degrees
    for degree in range(initial_degree, final_degree + 1, degree_step):
        #record computation time
        start_time = time.time()
        #do actual taylor approx
        _, approx_vals = taylor_approximation(func, start, end, degree, c)
        #compute the time elapsed
        elapsed = time.time() - start_time

        total_error = np.sum(np.abs(true_vals - approx_vals))

        #store results
        records.append({
            'degree': degree,
            'total_error': total_error,
            'time_seconds': elapsed
        })

    #convert to a pandas df
    df = pd.DataFrame(records)
    #save as a csv as asked in the assignment
    df.to_csv("taylor_values.csv", index=False)

    return df


def main():
    x = sp.symbols('x')
    #function definted by the assignment
    func = x * sp.sin(x)**2 + sp.cos(x)

    #also from the assignment
    start, end = -10, 10
    degree = 99
    c = 0

    #do the taylor approx
    x_vals, approx_vals = taylor_approximation(func, start, end, degree, c)

    #compare
    f_lambdified = sp.lambdify(x, func, modules='numpy')
    true_vals = f_lambdified(x_vals)

    #plot the function and taylor approx
    plt.figure(figsize=(15, 10))
    #black line similar to the assignment
    plt.plot(x_vals, true_vals, label="Actual", color = "black")
    #uses red dots similar to the assignment
    plt.plot(x_vals, approx_vals, "ro", label=f"Taylor Approximation")
    plt.xlabel("x")
    plt.ylabel("x*sin^2(x) + cos(x)")
    plt.title("Taylor Series Approximation of f(x) = x*sin^2(x) + cos(x)")
    plt.legend()
    plt.grid(True)
    plt.savefig("taylor_plot.png", dpi=300)
    plt.show()

    #do for multiple degrees
    df = factorial_analysis(func, start, end, c,
                            initial_degree=50,
                            final_degree=100,
                            degree_step=10)

    print(df)

#run above code
if __name__ == "__main__":
    main()